import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as T


def ensure_root_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(backbone: str, num_classes: int) -> nn.Module:
    ensure_root_on_path()
    if backbone == 'mcpx_no_attn':
        from model_mcpx import MetaCortexNet_NoAttn
        return MetaCortexNet_NoAttn(num_classes=num_classes)
    elif backbone == 'resnet152v2':
        import torchvision.models as models
        m = models.resnet152(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif backbone == 'mobilenet_v2':
        import torchvision.models as models
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif backbone == 'vgg16_bn':
        import torchvision.models as models
        m = models.vgg16_bn(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    elif backbone == 'vit_base':
        import torchvision.models as models
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    else:
        raise ValueError(backbone)


def load_weights_if_exist(model: nn.Module, weight_paths: List[str]) -> str:
    for p in weight_paths:
        if os.path.isfile(p):
            ckpt = torch.load(p, map_location='cpu')
            state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered)
            model.load_state_dict(model_dict)
            return f"{os.path.basename(p)} loaded_layers={len(filtered)}/{len(state)}"
    return ""


def discover_images(max_n: int = 32) -> List[str]:
    roots = [
        'rice leaf diseases dataset',
        'Rice Leaf Diseases Dataset/Rice Leaf Diseases Dataset/rice leaf diseases dataset/rice leaf diseases dataset',
    ]
    files = []
    for r in roots:
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp'):
            files.extend(glob.glob(os.path.join(r, '**', ext), recursive=True))
    return files[:max_n]


def apply_corruption(img: Image.Image, kind: str, severity: int) -> Image.Image:
    # simple approximations for common corruptions
    if kind == 'brightness':
        factor = 1.0 + (severity - 3) * 0.2  # sev=1..5 -> 0.6..1.4
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if kind == 'contrast':
        mean = 127.5
        factor = 1.0 + (severity - 3) * 0.25
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if kind == 'gaussian_blur':
        radius = 0.5 * severity
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    if kind == 'jpeg':
        quality = max(10, 100 - severity * 15)
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    if kind == 'gaussian_noise':
        sigma = severity * 5
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img


def to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tr = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tr(img)


def measure_pure_model_fps(model: nn.Module, inputs: torch.Tensor, warmup: int = 5, iters: int = 30, device: str = 'cpu') -> float:
    model.eval()
    with torch.no_grad():
        if device.startswith('cuda') and torch.cuda.is_available():
            for _ in range(warmup):
                _ = model(inputs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            for _ in range(warmup):
                _ = model(inputs)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(inputs)
            t1 = time.perf_counter()
    return iters / (t1 - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda or cuda:0')
    parser.add_argument('--max_images', type=int, default=16)
    parser.add_argument('--out_json', type=str, default='unified_benchmark/robustness_results.json')
    parser.add_argument('--out_md', type=str, default='unified_benchmark/robustness_results.md')
    args = parser.parse_args()

    os.makedirs('unified_benchmark', exist_ok=True)
    out_json = args.out_json
    out_md = args.out_md

    # device resolve
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    backbones = ['mcpx_no_attn', 'mobilenet_v2', 'resnet152v2', 'vgg16_bn']
    num_classes = 3
    input_sizes = [128, 224, 320]
    corruptions = ['brightness', 'contrast', 'gaussian_blur', 'jpeg', 'gaussian_noise']
    severities = [1, 3, 5]

    weights_map = {
        'mcpx_no_attn': [
            'runs_fixed/full_run/mcpx_no_attn/finetuned_mcpx_fixed.pth',
            'rice_leaf_transfer_results/mcpx_no_attn_final.pth'
        ],
        'mobilenet_v2': ['rice_leaf_transfer_results/mobilenet_v2_final.pth'],
        'resnet152v2': ['rice_leaf_transfer_results/resnet152v2_final.pth'],
        'vgg16_bn': ['rice_leaf_transfer_results/vgg16_final.pth', 'runs_fixed/full_run/vgg16/finetuned_mcpx_fixed.pth']
    }

    images = discover_images(max_n=args.max_images)

    results: Dict[str, Any] = {}
    for bk in backbones:
        model = build_model(bk, num_classes)
        model = model.to(device)
        note = load_weights_if_exist(model, weights_map.get(bk, []))
        results[bk] = {"weights_note": note, "entries": [], "device": device}
        for size in input_sizes:
            # prepare batch tensor once per corruption severity (N images)
            base_tensors = [to_tensor(Image.open(p).convert('RGB'), size) for p in images]
            base_batch = torch.stack(base_tensors, dim=0).to(device)
            fps_clean = measure_pure_model_fps(model, base_batch, device=device)
            results[bk]["entries"].append({
                "input": size,
                "corruption": "clean",
                "severity": 0,
                "pure_model_fps": fps_clean
            })
            for c in corruptions:
                for s in severities:
                    tensors = [to_tensor(apply_corruption(Image.open(p).convert('RGB'), c, s), size) for p in images]
                    batch = torch.stack(tensors, dim=0).to(device)
                    fps = measure_pure_model_fps(model, batch, device=device)
                    results[bk]["entries"].append({
                        "input": size,
                        "corruption": c,
                        "severity": s,
                        "pure_model_fps": fps
                    })

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    # markdown summary (per backbone, average FPS drop at size=224)
    lines = ["| Backbone | Corruption | Severity | Input | PureModelFPS |\n",
             "|---|---|---:|---:|---:|\n"]
    for bk, obj in results.items():
        for e in obj["entries"]:
            lines.append(f"| {bk} | {e['corruption']} | {e['severity']} | {e['input']} | {e['pure_model_fps']:.2f} |\n")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"Saved: {out_json}, {out_md}")


if __name__ == '__main__':
    main()


