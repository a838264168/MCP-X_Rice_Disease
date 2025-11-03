import argparse
import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


def ensure_root_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(backbone: str, num_classes: int) -> nn.Module:
    ensure_root_on_path()
    if backbone == 'mcpx_no_attn':
        from model_mcpx import MetaCortexNet_NoAttn
        return MetaCortexNet_NoAttn(num_classes=num_classes)
    elif backbone == 'mobilenet_v2':
        import torchvision.models as models
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif backbone == 'resnet152v2':
        import torchvision.models as models
        m = models.resnet152(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
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


def nvidia_smi_query_power() -> float:
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'] )
        vals = out.decode().strip().splitlines()
        if not vals:
            return float('nan')
        # take first GPU
        return float(vals[0])  # watts
    except Exception:
        return float('nan')


def measure_energy_gpu(model: nn.Module, inputs: torch.Tensor, iters: int = 100, device: str = 'cuda') -> Dict[str, Any]:
    model.eval()
    energies = []
    with torch.no_grad():
        # warmup
        for _ in range(10):
            _ = model(inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e0 = nvidia_smi_query_power()
        for _ in range(iters):
            _ = model(inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e1 = nvidia_smi_query_power()
    # approximate energy by average power * duration (J = W * s)
    # note: nvidia-smi sampling is sparse; this is a coarse lower bound
    avg_power_w = (e0 + e1) / 2 if (not (torch.isnan(torch.tensor(e0)) or torch.isnan(torch.tensor(e1)))) else float('nan')
    duration_s = (t1 - t0)
    energy_j = avg_power_w * duration_s if avg_power_w == avg_power_w else float('nan')
    return {"avg_power_w": float(avg_power_w), "duration_s": float(duration_s), "energy_j": float(energy_j)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--backbones', type=str, default='mcpx_no_attn,mobilenet_v2,resnet152v2,vgg16_bn')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--input', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--output', type=str, default='unified_benchmark/energy_results.json')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    weights_map = {
        'mcpx_no_attn': [
            'runs_fixed/full_run/mcpx_no_attn/finetuned_mcpx_fixed.pth',
            'rice_leaf_transfer_results/mcpx_no_attn_final.pth'
        ],
        'mobilenet_v2': ['rice_leaf_transfer_results/mobilenet_v2_final.pth'],
        'resnet152v2': ['rice_leaf_transfer_results/resnet152v2_final.pth'],
        'vgg16_bn': ['rice_leaf_transfer_results/vgg16_final.pth', 'runs_fixed/full_run/vgg16/finetuned_mcpx_fixed.pth']
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results: Dict[str, Any] = {}

    backbones = args.backbones.split(',')
    for bk in backbones:
        model = build_model(bk, args.num_classes).to(device)
        note = load_weights_if_exist(model, weights_map.get(bk, []))
        dummy = torch.randn(args.batch, 3, args.input, args.input, device=device)

        entry = {"device": device, "weights_note": note}
        if device.startswith('cuda') and torch.cuda.is_available():
            entry.update(measure_energy_gpu(model, dummy, iters=args.iters, device=device))
        else:
            entry.update({"avg_power_w": None, "duration_s": None, "energy_j": None, "note": "No GPU/RAPL available"})

        results[bk] = entry

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"Saved energy results to {args.output}")


if __name__ == '__main__':
    main()



