import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model_mcpx import MetaCortexNet


def build_model(backbone: str, num_classes: int):
    backbone = backbone.lower()
    if backbone == 'mcpx_no_attn':
        model = MetaCortexNet(num_classes=num_classes)
        if hasattr(model, 'disable_attention'):
            model.disable_attention()
        return model
    import torchvision.models as tvm
    if backbone == 'vit_base':
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    if backbone in ('resnet152v2', 'resnet152'):
        m = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if backbone == 'mobilenet_v2':
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V2)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if backbone == 'vgg16':
        m = tvm.vgg16_bn(weights=tvm.VGG16_BN_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f'Unknown backbone: {backbone}')


def get_tta_transforms(img_size: int):
    base = [
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return [
        transforms.Compose(base),
        transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 1.25)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 1.05)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(int(img_size * 1.25)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]


def predict_with_tta(model: nn.Module, loader: DataLoader, device, tta_transforms):
    model.eval()
    all_preds = []
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for x, _ in loader:
            batch_preds = []
            # x is an unnormalized tensor [0,1] here
            for tta in tta_transforms:
                x_tta = torch.stack([tta(to_pil(img)) for img in x])
                x_tta = x_tta.to(device, non_blocking=True)
                logits = model(x_tta)
                probs = torch.softmax(logits, dim=1)
                batch_preds.append(probs)
            avg_probs = torch.stack(batch_preds, dim=0).mean(0)
            all_preds.append(avg_probs)
    return torch.cat(all_preds, dim=0)


def load_model_with_temp(model_path: Path, backbone: str, num_classes: int, device):
    model = build_model(backbone, num_classes).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Temperature scaling
    T = 1.0
    temp_path = model_path.parent / 'temperature.json'
    if temp_path.exists():
        try:
            data = json.loads(temp_path.read_text(encoding='utf-8'))
            T = float(data.get('T', 1.0))
            print(f"Loaded temperature scaling T={T:.4f} for {model_path.name}")
        except Exception as e:
            print(f"Failed to load temperature scaling: {e}, fallback T=1.0")

    # Optional vector scaling: vector_scaling.json with scale/bias or W/b
    scale_vec = None
    bias_vec = None
    vs_path = model_path.parent / 'vector_scaling.json'
    if vs_path.exists():
        try:
            vs = json.loads(vs_path.read_text(encoding='utf-8'))
            if 'scale' in vs and isinstance(vs['scale'], list) and len(vs['scale']) == num_classes:
                scale_vec = torch.tensor(vs['scale'], dtype=torch.float32, device=device)
            elif 'W' in vs:
                W = vs['W']
                if isinstance(W, list) and len(W) == num_classes and not isinstance(W[0], list):
                    scale_vec = torch.tensor(W, dtype=torch.float32, device=device)
                elif isinstance(W, list) and len(W) == num_classes and isinstance(W[0], list):
                    # If a matrix is provided, take its diagonal as scaling vector
                    diag = [float(W[i][i]) for i in range(min(num_classes, len(W)))]
                    scale_vec = torch.tensor(diag, dtype=torch.float32, device=device)
            if 'bias' in vs and isinstance(vs['bias'], list) and len(vs['bias']) == num_classes:
                bias_vec = torch.tensor(vs['bias'], dtype=torch.float32, device=device)
            elif 'b' in vs and isinstance(vs['b'], list) and len(vs['b']) == num_classes:
                bias_vec = torch.tensor(vs['b'], dtype=torch.float32, device=device)
            if scale_vec is not None or bias_vec is not None:
                print(f"Loaded vector scaling for {model_path.name}")
        except Exception as e:
            print(f"Failed to load vector scaling: {e}, skip")

    class CalibratedModel(nn.Module):
        def __init__(self, mdl, T, scale_vec=None, bias_vec=None):
            super().__init__()
            self.mdl = mdl
            self.T = T
            self.register_buffer('scale_vec', scale_vec if scale_vec is not None else torch.ones(num_classes, dtype=torch.float32))
            self.register_buffer('bias_vec', bias_vec if bias_vec is not None else torch.zeros(num_classes, dtype=torch.float32))
        def forward(self, x):
            logits = self.mdl(x)
            # Ensure buffers are on the same device as logits
            scale = self.scale_vec.to(logits.device)
            bias = self.bias_vec.to(logits.device)
            logits = logits * scale + bias
            return logits / self.T

    return CalibratedModel(model, T, scale_vec, bias_vec)


def compute_entropy(prob: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    p = prob.clamp(min=eps)
    return -(p * p.log()).sum(dim=1)


def fuse_probs(probs_list, weights, mode: str) -> torch.Tensor:
    # probs_list: list of [B,C]
    if mode == 'mean':
        out = torch.zeros_like(probs_list[0])
        for i, p in enumerate(probs_list):
            out += weights[i] * p
        return out
    if mode == 'geom':
        eps = 1e-12
        log_sum = torch.zeros_like(probs_list[0])
        for i, p in enumerate(probs_list):
            log_sum += weights[i] * (p.clamp(min=eps).log())
        return (log_sum).exp()
    if mode == 'rank':
        # rank larger is better: score = (C - rank)
        B, C = probs_list[0].shape
        score = torch.zeros_like(probs_list[0])
        for i, p in enumerate(probs_list):
            # ranks: argsort descending → positions
            sorted_idx = torch.argsort(p, dim=1, descending=True)
            ranks = torch.zeros_like(sorted_idx)
            ranks.scatter_(1, sorted_idx, torch.arange(C, device=p.device).unsqueeze(0).expand(B, C))
            inv = (C - ranks.float())
            score += weights[i] * inv
        # convert to pseudo-prob by softmax to keep scale similar
        return torch.softmax(score, dim=1)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(description='Ensemble evaluation with temperature scaling and TTA')
    ap.add_argument('--data_root', type=str, required=True, help='ImageFolder root (test split) or a split root with test/')
    ap.add_argument('--models', type=str, nargs='+', required=True, help='List of model .pth paths')
    ap.add_argument('--backbones', type=str, nargs='+', help='List of backbone names corresponding to models')
    ap.add_argument('--out_dir', type=str, default='ensemble_outputs')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--weights', type=float, nargs='+', help='Optional weights for weighted ensemble')
    ap.add_argument('--fusion', type=str, default='mean', choices=['mean','geom','rank'])
    ap.add_argument('--use_tta', action='store_true', help='Enable Test-Time Augmentation')
    ap.add_argument('--tta_transforms', type=int, default=5, help='Number of TTA transforms to use')
    ap.add_argument('--use_tta_gated', action='store_true', help='Use entropy-gated TTA (compute no-TTA first, then TTA for high-entropy samples)')
    ap.add_argument('--entropy_threshold', type=float, default=1.0, help='Entropy threshold for gated TTA')
    ap.add_argument('--tta_batch', type=int, default=32, help='Mini-batch size for gated TTA subset')
    ap.add_argument('--multi_res', type=int, nargs='*', default=[], help='Multiple resolutions (e.g., 224 256 288) for no-TTA multi-res fusion')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_root = Path(args.data_root)
    if (test_root / 'test').exists():
        test_root = test_root / 'test'

    # Dataset and dataloader
    if args.use_tta:
        # Keep [0,1] without Normalize for ToPILImage TTA pipeline
        tf = transforms.Compose([
            transforms.Resize(int(args.img_size * 1.15)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
        ])
        batch_size = 32
    else:
        tf = transforms.Compose([
            transforms.Resize(int(args.img_size * 1.15)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        batch_size = 64

    ds = datasets.ImageFolder(test_root, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    num_classes = len(ds.classes)

    # Infer backbone names
    if args.backbones:
        backbones = args.backbones
    else:
        backbones = []
        for p in args.models:
            s = str(p).lower()
            if 'mcpx' in s:
                backbones.append('mcpx_no_attn')
            elif 'vit' in s:
                backbones.append('vit_base')
            elif 'resnet' in s:
                backbones.append('resnet152v2')
            elif 'mobilenet' in s:
                backbones.append('mobilenet_v2')
            elif 'vgg' in s:
                backbones.append('vgg16')
            else:
                backbones.append('mcpx_no_attn')

    # Load models
    models = [load_model_with_temp(Path(p), b, num_classes, device) for p, b in zip(args.models, backbones)]

    # Weights
    if args.weights:
        weights = torch.tensor(args.weights, device=device)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(models), device=device) / len(models)

    print(f"Ensembling {len(models)} models, weights: {weights.cpu().numpy()}")
    print(f"Fusion: {args.fusion}")

    class_names = ds.classes
    y_true, y_pred = [], []

    # no-TTA pass probs for gating
    probs_no_tta = None
    if args.use_tta_gated:
        with torch.no_grad():
            all_probs = []
            for x, y in loader:
                x = x.to(device)
                probs_list = [torch.softmax(m(x), dim=1) for m in models]
                fused = fuse_probs(probs_list, weights, args.fusion)
                all_probs.append(fused.cpu())
            probs_no_tta = torch.cat(all_probs, dim=0)  # [N,C]
        ent = compute_entropy(probs_no_tta)
        idx_h = (ent >= args.entropy_threshold).nonzero(as_tuple=False).squeeze(1)
        print(f"Entropy-gated TTA: high-entropy samples {int(idx_h.numel())}/{len(ds)} will use TTA")

    # predict
    if args.use_tta or args.use_tta_gated:
        # prepare TTA list
        def get_tta_transforms(img_size: int):
            base = [transforms.Resize(int(img_size * 1.15)), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
            return [
                transforms.Compose(base),
                transforms.Compose([transforms.Resize(int(img_size*1.15)),transforms.CenterCrop(img_size),transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                transforms.Compose([transforms.Resize(int(img_size*1.25)),transforms.CenterCrop(img_size),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                transforms.Compose([transforms.Resize(int(img_size*1.05)),transforms.CenterCrop(img_size),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                transforms.Compose([transforms.Resize(int(img_size*1.25)),transforms.CenterCrop(img_size),transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
            ]
        tta_list = get_tta_transforms(args.img_size)

        # collect dataset tensors for TTA reuse
        imgs, labels = [], []
        for x, y in loader:
            imgs.append(x.cpu()); labels.extend(y.numpy().tolist())
        imgs = torch.cat(imgs, dim=0)  # CPU tensor [N,3,H,W]

        if args.use_tta and not args.use_tta_gated:
            # Full TTA for all samples
            with torch.no_grad():
                all_model_probs_tta = []
                for m in models:
                    probs_accum = []
                    for t in tta_list[:args.tta_transforms]:
                        xt = torch.stack([t(transforms.ToPILImage()(img)) for img in imgs])
                        xt = xt.to(device)
                        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                            probs_accum.append(torch.softmax(m(xt), dim=1))
                    probs_tta = torch.stack(probs_accum, dim=0).mean(0)
                    all_model_probs_tta.append(probs_tta)
                fused = fuse_probs(all_model_probs_tta, weights, args.fusion)
                y_pred = torch.argmax(fused, dim=1).cpu().numpy().tolist()
                y_true = labels
        else:
            # Entropy-gated TTA on high-entropy subset only (mini-batch)
            assert probs_no_tta is not None
            fused_no = probs_no_tta.to(device)
            fused = fused_no.clone()
            bs = max(1, int(args.tta_batch))
            with torch.no_grad():
                for start in range(0, idx_h.numel(), bs):
                    sel = idx_h[start:start+bs]
                    mb_model_probs = []
                    for m in models:
                        probs_accum = []
                        for t in tta_list[:args.tta_transforms]:
                            xt = torch.stack([t(transforms.ToPILImage()(imgs[i])) for i in sel.tolist()])
                            xt = xt.to(device)
                            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                                probs_accum.append(torch.softmax(m(xt), dim=1))
                        probs_tta_mb = torch.stack(probs_accum, dim=0).mean(0)
                        mb_model_probs.append(probs_tta_mb)
                    fused_mb = fuse_probs(mb_model_probs, weights, args.fusion)
                    fused[sel] = fused_mb
            y_pred = torch.argmax(fused, dim=1).cpu().numpy().tolist()
            y_true = labels

    else:
        # No-TTA branch: support single and multi-resolution (average probs across resolutions)
        if args.multi_res:
            # Preload full dataset as CPU tensors to avoid repeated IO
            tf_raw = transforms.Compose([transforms.ToTensor()])
            ds_raw = datasets.ImageFolder(test_root, transform=tf_raw)
            imgs = torch.stack([ds_raw[i][0] for i in range(len(ds_raw))], dim=0)  # [N,3,H,W] in [0,1]
            labels = [ds_raw[i][1] for i in range(len(ds_raw))]
            with torch.no_grad():
                # For each model, forward at multiple resolutions and average per-model probs, then fuse across models
                per_model_probs = []  # list of [N,C] on CPU
                bs = 64
                for mdl, bname in zip(models, backbones):
                    # ViT strictly requires 224; others can use provided multi-res sizes
                    if bname.lower().startswith('vit'):
                        sizes = [224]
                    else:
                        sizes = args.multi_res
                    probs_sizes = []
                    for sz in sizes:
                        accum = torch.empty(0, num_classes, device='cpu')
                        for start in range(0, len(ds_raw), bs):
                            end = min(len(ds_raw), start + bs)
                            xb = imgs[start:end].clone()
                            # Tensor pipeline: F.resize -> F.center_crop -> F.normalize
                            x_list = []
                            for img in xb:
                                x = F.resize(img, int(sz * 1.15))
                                x = F.center_crop(x, [sz, sz])
                                x = F.normalize(x, [0.485,0.456,0.406], [0.229,0.224,0.225])
                                x_list.append(x)
                            xb = torch.stack(x_list, dim=0).to(device)
                            p = torch.softmax(mdl(xb), dim=1).cpu()
                            accum = torch.cat([accum, p], dim=0)
                        probs_sizes.append(accum)
                    # Per-model multi-resolution average
                    avg_model = torch.stack(probs_sizes, dim=0).mean(0)
                    per_model_probs.append(avg_model)
                # Fuse across models
                fused = fuse_probs([p.to(device) for p in per_model_probs], weights, args.fusion).cpu()
                y_pred = torch.argmax(fused, dim=1).numpy().tolist()
                y_true = labels
        else:
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    probs_list = [torch.softmax(m(x), dim=1) for m in models]
                    fused = fuse_probs(probs_list, weights, args.fusion)
                    y_pred.extend(torch.argmax(fused, dim=1).cpu().numpy().tolist())
                    y_true.extend(y.numpy().tolist())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics and visualization
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)), target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / 'ensemble_per_class_metrics.csv', encoding='utf-8')

    with open(out_dir / 'ensemble_per_class_metrics.md', 'w', encoding='utf-8') as f:
        f.write("# Ensemble Classification Report\n\n")
        f.write(f"**Overall Accuracy**: {report['accuracy']:.4f}\n\n")
        f.write(f"**Fusion**: {args.fusion}\n\n")
        if args.use_tta:
            f.write(f"**TTA**: {args.tta_transforms}\n\n")
        if args.use_tta_gated:
            f.write(f"**TTA gated**, entropy_threshold={args.entropy_threshold}\n\n")
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|----------|\n")
        for name in class_names:
            if name in report:
                m = report[name]
                f.write(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {m['support']} |\n")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / 'ensemble_confusion_matrix.csv', encoding='utf-8')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Ensemble Confusion Matrix (Raw) - {args.fusion}")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    plt.savefig(out_dir / 'ensemble_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(out_dir / 'ensemble_confusion_matrix_normalized.csv', encoding='utf-8')

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.3f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Ensemble Confusion Matrix (Normalized) - {args.fusion}")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
    plt.savefig(out_dir / 'ensemble_confusion_matrix_normalized.png', dpi=200, bbox_inches='tight')
    plt.close()

    meta_info = {
        'models': args.models,
        'backbones': backbones,
        'weights': weights.cpu().numpy().tolist(),
        'num_classes': num_classes,
        'overall_accuracy': report['accuracy'],
        'fusion': args.fusion,
        'tta_enabled': args.use_tta,
        'tta_gated': args.use_tta_gated,
        'entropy_threshold': args.entropy_threshold if args.use_tta_gated else None
    }
    with open(out_dir / 'ensemble_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    print(f"Ensemble evaluation completed. Accuracy: {report['accuracy']:.4f}")
    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()


