# Paper Reproduction Code (Rice Leaf & Unified Benchmark)

Environment (exact, reproducible):
- Use only `environment.yml` (conda):

```
conda env create -f environment.yml
conda activate tf_env
```

Note: No pip requirements.txt is provided; always use `environment.yml`.

## Datasets
- Set `--data_root` to your dataset root (ImageFolder structure) for Rice Leaf Diseases.

## Train / Evaluate
```
python train.py --data_root "PATH_TO_RICE_LEAF_DATASET" --use_transfer_learning --epochs 20 --batch_size 64
```

## Unified Benchmark
```
python unified_benchmark.py --models mcpx_no_attn,mobilenet_v2,resnet152v2,vgg16_bn,vit_base --num_classes 3 --input_size 224
```

## Robustness
```
python robustness_eval.py --device auto --max_images 16
```

## Energy (GPU)
```
python energy_eval.py --device auto --backbones mcpx_no_attn,mobilenet_v2,resnet152v2,vgg16_bn --input 224 --batch 32
```

## Grad-CAM
```
python gradcam_generate.py
```

## Ensemble Learning (historical best script)
```
python ensemble_eval.py --data_root "PATH_TO_TEST_OR_SPLIT_ROOT" \
  --models runs_fixed/vit_base_e100/finetuned_mcpx_fixed.pth \
           runs_fixed/resnet152v2_e100/finetuned_mcpx_fixed.pth \
           runs_fixed/mobilenet_v2_e100/finetuned_mcpx_fixed.pth \
           runs_fixed/vgg16_e100/finetuned_mcpx_fixed.pth \
           runs_fixed/mcpx_no_attn_e100/finetuned_mcpx_fixed.pth \
  --out_dir ensemble_outputs --workers 0 --fusion mean
```

Note:
- This repository only provides `ensemble_eval.py` for ensembling.

