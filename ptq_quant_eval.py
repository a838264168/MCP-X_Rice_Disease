import argparse
import os
import sys
import time
import json
from typing import Dict, Any

import torch
import torch.nn as nn


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


def load_weights_if_exist(model: nn.Module, weight_paths):
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


def count_params_and_buffers(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())


def memory_usage_mb(model: nn.Module, dtype_bytes: int = 4) -> float:
    return count_params_and_buffers(model) * dtype_bytes / (1024 ** 2)


def measure_fps_cpu(model: nn.Module, input_tensor: torch.Tensor, warmup: int = 10, iters: int = 100) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(input_tensor)
        t1 = time.perf_counter()
    return iters / (t1 - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbones', type=str, default='mcpx_no_attn,mobilenet_v2,resnet152v2,vgg16_bn')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--input', type=int, default=224)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', type=str, default='unified_benchmark/ptq_quant_results.json')
    args = parser.parse_args()

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

    for bk in args.backbones.split(','):
        # fp32 model
        m_fp32 = build_model(bk, args.num_classes)
        note = load_weights_if_exist(m_fp32, weights_map.get(bk, []))
        dummy = torch.randn(args.batch, 3, args.input, args.input)
        fps_fp32 = measure_fps_cpu(m_fp32, dummy)
        mem_fp32 = memory_usage_mb(m_fp32, dtype_bytes=4)

        # dynamic quantization (CPU linear layers)
        m_int8 = torch.quantization.quantize_dynamic(
            m_fp32, {nn.Linear}, dtype=torch.qint8
        )
        fps_int8 = measure_fps_cpu(m_int8, dummy)
        # weights become int8 for linear layers (approx. lower bound on memory)
        mem_int8 = memory_usage_mb(m_int8, dtype_bytes=1)

        results[bk] = {
            'weights_note': note,
            'batch': args.batch,
            'input': args.input,
            'fps_fp32': fps_fp32,
            'fps_int8': fps_int8,
            'memory_fp32_mb': mem_fp32,
            'memory_int8_mb_approx': mem_int8,
        }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved PTQ results to {args.output}")


if __name__ == '__main__':
    main()



