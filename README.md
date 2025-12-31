# Paper Reproduction Code (Rice Leaf & Unified Benchmark)
Abstract

Rice, a dietary staple for over half of the global population, is highly susceptible to bacterial and fungal diseases such as bacterial blight, brown spot, and leaf smut, which can severely reduce yields. Traditional manual detection is labor-intensive and often results in delayed intervention and excessive chemical use. Although deep learning models like convolutional neural networks (CNNs) achieve high accuracy, their computational demands hinder deployment in resource-limited agricultural settings. We propose MCP-X, an ultra-compact CNN with only 0.21 million parameters for real-time, on-device rice disease classification. MCP-X integrates a shallow encoder, multi-branch expert routing, a bi-level recurrent simulation encoder–decoder (BRSE), an efficient channel attention (ECA) module, and a lightweight classifier. Trained from scratch, MCP-X achieves 98.93% accuracy on PlantVillage and 96.59% on the Rice Disease Detection Dataset, without external pretraining. Mechanistically, expert routing diversifies feature branches, ECA enhances channel-wise signal relevance, and BRSE captures lesion-scale and texture cues—yielding complementary, stage-wise gains confirmed through ablation studies. Despite slightly higher FLOPs than MobileNetV2, MCP-X prioritizes a minimal memory footprint (~1.01 MB) and deployability over raw speed, running at 53.83 FPS (2.42 GFLOPs) on an RTX A5000. It achieves 16.7×, 287×, 420×, and 659× fewer parameters than MobileNetV2, ResNet152V2, ViT-Base, and VGG-16, respectively. When integrated into a multi-resolution ensemble, MCP-X attains 99.85% accuracy, demonstrating exceptional robustness across controlled and field datasets while maintaining efficiency for real-world agricultural applications.

Keywords: rice disease classification; convolutional neural network; lightweight model; efficient channel attention; PlantVillage dataset; resource-constrained deployment

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

