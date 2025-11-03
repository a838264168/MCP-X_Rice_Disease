#!/usr/bin/env python3
"""
Rice Leaf Diseases training/evaluation script – supports transfer learning and training from scratch.
Deterministic seeds; saves all outputs and experiment metadata for reproducibility.
"""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import copy
import sys

# Import model definitions
from model_mcpx import MetaCortexNet, MetaCortexNet_NoAttn, BaselineNet, MCPX


def set_seed(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed fixed for reproducibility: {seed}")


def build_dataloaders(data_root: str, img_size: int = 224, batch_size: int = 64, workers: int = 0, train_split: float = 0.8, val_split: float = 0.1):
    """Build dataloaders with train/val/test splits."""
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(data_root, transform=train_transform)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    # Split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set eval transforms for val/test
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    
    # DataLoaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=(workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, persistent_workers=(workers > 0)
    )
    
    print(f"📊 Dataset:")
    print(f"  - Total: {total_size}")
    print(f"  - Train: {train_size} ({train_size/total_size:.1%})")
    print(f"  - Val: {val_size} ({val_size/total_size:.1%})")
    print(f"  - Test: {test_size} ({test_size/total_size:.1%})")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    
    return train_loader, val_loader, test_loader, num_classes, class_names


def build_model(backbone: str, num_classes: int):
    """Build model by backbone name."""
    backbone = backbone.lower()
    if backbone == 'mcpx_no_attn':
        model = MetaCortexNet_NoAttn(num_classes=num_classes)
    elif backbone == 'mcpx_full':
        model = MetaCortexNet(num_classes=num_classes)
    elif backbone == 'baseline':
        model = BaselineNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return model


def load_pretrained_model(model_path: str, backbone: str, num_classes: int, device, use_transfer_learning: bool = True):
    """Load weights for transfer learning or initialize from scratch."""
    print(f"🔄 Loading model: {model_path} ({backbone})")
    
    # Create model
    model = build_model(backbone, num_classes)
    
    if use_transfer_learning and Path(model_path).exists():
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Current model state dict
        model_state_dict = model.state_dict()
        
        # Filter out classifier layers due to class-mismatch
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                if 'fc_final' in key or 'classifier' in key or 'head' in key:
                    # skip classifier due to different num_classes
                    print(f"  ⚠️ Skip classifier layer: {key}")
                    continue
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"  ⚠️ Shape mismatch, skip: {key} (ckpt: {value.shape}, current: {model_state_dict[key].shape})")
            else:
                print(f"  ⚠️ Layer not found in current model: {key}")
        
        # Load filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ Transfer learning: loaded {len(filtered_state_dict)} layers")
    else:
        print(f"✅ Training from scratch: random initialization")
    
    model = model.to(device)
    return model


def train_model(model, train_loader, val_loader, device, epochs: int = 20, lr: float = 1e-4):
    """Train model and return best model by validation accuracy."""
    print(f"🏋️ Start training, epochs={epochs}, lr={lr}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Best Val Acc={best_val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Training complete, best val acc: {best_val_acc:.4f}")
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model on a dataloader and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_metrics = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'per_class_precision': per_class_metrics[0].tolist(),
        'per_class_recall': per_class_metrics[1].tolist(),
        'per_class_f1': per_class_metrics[2].tolist(),
        'class_names': class_names
    }
    
    return results


def save_results(results, output_dir, model_name, class_names):
    """Save results to files (JSON, confusion matrix image/CSV, markdown)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    json_path = output_dir / f"{model_name}_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = copy.deepcopy(results)
        if 'predictions' in json_results:
            json_results['predictions'] = [int(x) for x in json_results['predictions']]
        if 'labels' in json_results:
            json_results['labels'] = [int(x) for x in json_results['labels']]
        if 'probabilities' in json_results:
            json_results['probabilities'] = [prob.tolist() for prob in json_results['probabilities']]
        
        json.dump(json_results, f, indent=2, ensure_ascii=True)
    
    # Save confusion matrix CSV
    cm = confusion_matrix(results['labels'], results['predictions'])
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save per-class metrics CSV
    per_class_df = pd.DataFrame({
        'class': class_names,
        'precision': results['per_class_precision'],
        'recall': results['per_class_recall'],
        'f1_score': results['per_class_f1']
    })
    per_class_df.to_csv(output_dir / f"{model_name}_per_class_metrics.csv", index=False)
    
    # Save markdown report
    md_path = output_dir / f"{model_name}_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {model_name} Evaluation Report\n\n")
        f.write(f"**Evaluated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Metrics\n\n")
        f.write(f"- **Accuracy**: {results['accuracy']:.4f}\n")
        f.write(f"- **Precision**: {results['precision']:.4f}\n")
        f.write(f"- **Recall**: {results['recall']:.4f}\n")
        f.write(f"- **F1-score**: {results['f1_score']:.4f}\n\n")
        
        f.write(f"## Per-class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-score |\n")
        f.write("|------|-----------:|-------:|---------:|\n")
        for i, class_name in enumerate(class_names):
            f.write(f"| {class_name} | {results['per_class_precision'][i]:.4f} | "
                   f"{results['per_class_recall'][i]:.4f} | {results['per_class_f1'][i]:.4f} |\n")
    
    print(f"✅ Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Rice Leaf Diseases training/evaluation with transfer learning support')
    parser.add_argument('--data_root', type=str, 
                       default=r'C:\Users\wku\Documents\GitHub\riceleaftestdataset\rice leaf diseases dataset',
                       help='Path to rice leaf diseases dataset root')
    parser.add_argument('--output_dir', type=str, default='rice_leaf_transfer_results',
                       help='Output directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--use_transfer_learning', action='store_true', 
                       help='Enable transfer learning (load pretrained weights)')
    parser.add_argument('--test_only', action='store_true', 
                       help='Test only, no training')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Build dataloaders
    train_loader, val_loader, test_loader, num_classes, class_names = build_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.workers
    )
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config = {
        'data_root': args.data_root,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'use_transfer_learning': args.use_transfer_learning,
        'num_classes': num_classes,
        'class_names': class_names,
        'test_time': datetime.now().isoformat()
    }
    
    with open(output_dir / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=True)
    
    # Define models to evaluate
    models_to_test = [
        ('runs_fixed/full_run/mcpx_no_attn/finetuned_mcpx_fixed.pth', 'mcpx_no_attn', 'MCPX_NoAttn'),
        ('runs_fixed/full_run/vit_base/finetuned_mcpx_fixed.pth', 'vit_base', 'ViT_Base'),
        ('runs_fixed/full_run/resnet152v2/finetuned_mcpx_fixed.pth', 'resnet152v2', 'ResNet152V2'),
        ('runs_fixed/full_run/mobilenet_v2/finetuned_mcpx_fixed.pth', 'mobilenet_v2', 'MobileNetV2'),
        ('runs_fixed/full_run/vgg16/finetuned_mcpx_fixed.pth', 'vgg16', 'VGG16'),
    ]
    
    for model_path, backbone, model_name in models_to_test:
        try:
            # Load model
            model = load_pretrained_model(
                model_path, backbone, num_classes, device, 
                use_transfer_learning=args.use_transfer_learning
            )
            
            # Train (if not test-only)
            if not args.test_only:
                model = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
            
            # Test
            results = evaluate_model(model, test_loader, device, class_names)
            save_results(results, output_dir, model_name, class_names)
            
            print(f"📊 {model_name} results:")
            print(f"  - Accuracy: {results['accuracy']:.4f}")
            print(f"  - Precision: {results['precision']:.4f}")
            print(f"  - Recall: {results['recall']:.4f}")
            print(f"  - F1-score: {results['f1_score']:.4f}")
            print()
            
        except Exception as e:
            print(f"❌ Error while evaluating {model_name}: {e}")
    
    print(f"🎉 All done! Results saved under: {output_dir}")


if __name__ == '__main__':
    main()

