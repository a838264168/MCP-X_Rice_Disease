import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


def set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params_and_buffers(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    return total_params, total_buffers


def memory_usage_mb(model: nn.Module, dtype_bytes: int = 4) -> float:
    params, buffers = count_params_and_buffers(model)
    return (params + buffers) * dtype_bytes / (1024 ** 2)


def measure_fps(model: nn.Module, input_tensor: torch.Tensor, warmup: int = 10, iters: int = 100) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(input_tensor)
        # pure model inference (no dataload)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(input_tensor)
        t1 = time.perf_counter()
    pure_fps = iters / (t1 - t0)
    return {"pure_model_fps": pure_fps}


def try_thop_flops(model: nn.Module, input_tensor: torch.Tensor) -> float:
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        # FLOPs ~= 2 * MACs (convention differs; we report MACs->GFLOPs directly for consistency)
        gflops = macs / 1e9
        return float(gflops)
    except Exception:
        return float('nan')


def build_model(backbone: str, num_classes: int) -> nn.Module:
    # ensure project root in sys.path so we can import local modules like model_mcpx
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    if backbone == 'mcpx_no_attn':
        from model_mcpx import MetaCortexNet_NoAttn
        return MetaCortexNet_NoAttn(num_classes=num_classes)
    elif backbone == 'mcpx':
        from model_mcpx import MetaCortexNet
        return MetaCortexNet(num_classes=num_classes)
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
        raise ValueError(f"Unsupported backbone: {backbone}")


def load_weights_if_exist(model: nn.Module, weight_path: str) -> str:
    if not weight_path or not os.path.isfile(weight_path):
        return ""
    ckpt = torch.load(weight_path, map_location='cpu')
    state = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    return f"loaded_layers={len(filtered)}/{len(state)}"


def benchmark_once(backbone: str, num_classes: int, input_size: int, weights: str) -> Dict[str, Any]:
    model = build_model(backbone, num_classes)
    note = load_weights_if_exist(model, weights)
    dummy = torch.randn(1, 3, input_size, input_size)
    gflops = try_thop_flops(model, dummy)
    fps_dict = measure_fps(model, dummy)
    total_params, total_buffers = count_params_and_buffers(model)
    mem_mb = memory_usage_mb(model)
    result = {
        "backbone": backbone,
        "num_classes": num_classes,
        "input_size": input_size,
        "total_params": int(total_params),
        "total_buffers": int(total_buffers),
        "memory_usage_mb": mem_mb,
        "gflops": gflops,
        **fps_dict,
        "weights_note": note,
        "device": "cpu"
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, required=False,
                        default='mcpx_no_attn,mobilenet_v2,resnet152v2,vgg16_bn,vit_base')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--weights_dir', type=str, default='runs_fixed')
    parser.add_argument('--output', type=str, default='unified_benchmark/results_unified.json')
    args = parser.parse_args()

    set_deterministic(42)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Candidate weight file paths (load first that exists)
    candidate_weights = {
        'mcpx_no_attn': [
            'runs_fixed/full_run/mcpx_no_attn/finetuned_mcpx_fixed.pth',
            'rice_leaf_transfer_results/mcpx_no_attn_final.pth'
        ],
        'mcpx': [
            'runs_fixed/full_run/mcpx/finetuned_mcpx_fixed.pth'
        ],
        'mobilenet_v2': [
            'rice_leaf_transfer_results/mobilenet_v2_final.pth'
        ],
        'resnet152v2': [
            'rice_leaf_transfer_results/resnet152v2_final.pth'
        ],
        'vgg16_bn': [
            'rice_leaf_transfer_results/vgg16_final.pth',
            'runs_fixed/full_run/vgg16/finetuned_mcpx_fixed.pth'
        ],
        'vit_base': [
            'rice_leaf_transfer_results/vit_base_final.pth'
        ],
    }

    results: Dict[str, Any] = {}
    for backbone in args.models.split(','):
        weight_path = ''
        for p in candidate_weights.get(backbone, []):
            if os.path.isfile(p):
                weight_path = p
                break
        results[backbone] = benchmark_once(backbone, args.num_classes, args.input_size, weight_path)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"Saved unified results to {args.output}")


if __name__ == '__main__':
    main()


