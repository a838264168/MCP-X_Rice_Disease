import os
import sys
import glob
import json
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
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
    else:
        raise ValueError(backbone)


def load_weights(model: nn.Module, path: str):
    if not path or not os.path.isfile(path):
        return
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()
        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_backward_hook(bwd_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()
        grads = self.gradients  # (N, C, H, W)
        acts = self.activations
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)  # (N, H, W)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    img_np = np.asarray(img).astype(np.float32) / 255.0
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((img.width, img.height))) / 255.0
    heatmap = np.zeros_like(img_np)
    heatmap[..., 0] = cam_resized  # simple red channel heat
    out = (1 - alpha) * img_np + alpha * heatmap
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def pick_target_layer(model: nn.Module) -> nn.Module:
    # heuristics for common CNNs
    for n, m in reversed(list(model.named_modules())):
        if isinstance(m, nn.Conv2d):
            return m
    # fallback to any last module
    modules = list(model.modules())
    return modules[-1]


def load_images(max_n: int = 8) -> list:
    candidates = []
    roots = [
        'rice leaf diseases dataset',
        'Rice Leaf Diseases Dataset/Rice Leaf Diseases Dataset/rice leaf diseases dataset/rice leaf diseases dataset',
    ]
    for r in roots:
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp'):
            candidates.extend(glob.glob(os.path.join(r, '**', ext), recursive=True))
    return candidates[:max_n]


def main():
    os.makedirs('interpretability_outputs', exist_ok=True)
    device = 'cpu'
    backbones = ['mcpx_no_attn', 'mobilenet_v2', 'resnet152v2', 'vgg16_bn']
    num_classes = 3

    weights_map = {
        'mcpx_no_attn': [
            'runs_fixed/full_run/mcpx_no_attn/finetuned_mcpx_fixed.pth',
            'rice_leaf_transfer_results/mcpx_no_attn_final.pth'
        ],
        'mobilenet_v2': ['rice_leaf_transfer_results/mobilenet_v2_final.pth'],
        'resnet152v2': ['rice_leaf_transfer_results/resnet152v2_final.pth'],
        'vgg16_bn': ['rice_leaf_transfer_results/vgg16_final.pth', 'runs_fixed/full_run/vgg16/finetuned_mcpx_fixed.pth']
    }

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_paths = load_images()
    for bk in backbones:
        model = build_model(bk, num_classes)
        # load first existing weights
        for wp in weights_map.get(bk, []):
            if os.path.isfile(wp):
                load_weights(model, wp)
                break
        target_layer = pick_target_layer(model)
        cam = GradCAM(model, target_layer)

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                continue
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                _ = model(input_tensor)
            heat = cam.generate(input_tensor)
            out_img = overlay_cam_on_image(img, heat)
            out_path = os.path.join('interpretability_outputs', f'{bk}_{i}.png')
            out_img.save(out_path)

    # record meta
    meta = {
        'device': device,
        'num_classes': num_classes,
        'backbones': backbones,
        'num_images': len(image_paths)
    }
    with open('interpretability_outputs/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)


if __name__ == '__main__':
    main()



