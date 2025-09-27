import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model.CellApop import CellApop
from utils.fine_tune_dataset import Labeled_CellDataset
from torch.utils.data import DataLoader
from utils.metrics import dice_coefficient

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = defaultdict(list)
        self.class_names = ['Background', 'Normal Cell', 'Apoptotic Cell']
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """加载模型"""
        print(f"Loading model from {self.config.model_path}")
        model = CellApop(
            img_size=self.config.img_size, 
            num_classes=self.config.num_classes, 
            encoder_fuse=True, 
            freeze=True
        ).to(self.device)
        
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def calculate_advanced_metrics(self, pred, target, class_idx=None):
        """计算高级评估指标"""
        metrics = {}
        
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred
            
        if torch.is_tensor(target):
            target_np = target.cpu().numpy()
        else:
            target_np = target
        
        if class_idx is not None:
            pred_binary = (pred_np == class_idx).astype(np.uint8)
            target_binary = (target_np == class_idx).astype(np.uint8)
        else:
            pred_binary = pred_np.astype(np.uint8)
            target_binary = target_np.astype(np.uint8)
        
        # 基础指标
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        # Dice系数
        dice = (2.0 * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
        metrics['dice'] = dice
        
        # IoU (Jaccard Index)
        iou = intersection / (union + 1e-8)
        metrics['iou'] = iou
        
        # 精确度和召回率
        tp = intersection
        fp = pred_binary.sum() - intersection
        fn = target_binary.sum() - intersection
        tn = pred_binary.size - tp - fp - fn
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
        
        # Hausdorff距离（仅对二值图像）
        if pred_binary.sum() > 0 and target_binary.sum() > 0:
            try:
                # 获取边界点
                pred_points = np.column_stack(np.where(pred_binary))
                target_points = np.column_stack(np.where(target_binary))
                
                if len(pred_points) > 0 and len(target_points) > 0:
                    hd1 = directed_hausdorff(pred_points, target_points)[0]
                    hd2 = directed_hausdorff(target_points, pred_points)[0]
                    hd = max(hd1, hd2)
                    metrics['hausdorff_distance'] = hd
                else:
                    metrics['hausdorff_distance'] = float('inf')
            except:
                metrics['hausdorff_distance'] = float('inf')
        else:
            metrics['hausdorff_distance'] = float('inf')
        
        # 平均表面距离
        if pred_binary.sum() > 0 and target_binary.sum() > 0:
            try:
                pred_surface = self._get_surface_points(pred_binary)
                target_surface = self._get_surface_points(target_binary)
                
                if len(pred_surface) > 0 and len(target_surface) > 0:
                    asd = self._average_surface_distance(pred_surface, target_surface)
                    metrics['average_surface_distance'] = asd
                else:
                    metrics['average_surface_distance'] = float('inf')
            except:
                metrics['average_surface_distance'] = float('inf')
        else:
            metrics['average_surface_distance'] = float('inf')
        
        return metrics
    
    def _get_surface_points(self, binary_mask):
        """获取表面点"""
        # 使用形态学操作获取边界
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
        boundary = binary_mask.astype(np.uint8) - eroded
        return np.column_stack(np.where(boundary))
    
    def _average_surface_distance(self, surface1, surface2):
        """计算平均表面距离"""
        from scipy.spatial.distance import cdist
        distances1 = cdist(surface1, surface2).min(axis=1)
        distances2 = cdist(surface2, surface1).min(axis=1)
        return (distances1.mean() + distances2.mean()) / 2
    
    def evaluate_model(self, model):
        """评估模型"""
        print("Starting model evaluation...")
        
        # 加载测试数据
        test_dataset = Labeled_CellDataset(self.config.test_json, 'test')
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        all_predictions = []
        all_targets = []
        sample_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                start_time = time.time()
                outputs, features = model(images)
                inference_time = time.time() - start_time
                
                # 获取预测结果
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # 保存结果用于后续分析
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(labels.cpu().numpy().flatten())
                
                # 计算每个样本的指标
                for i in range(images.size(0)):
                    sample_pred = predictions[i].cpu().numpy()
                    sample_target = labels[i].cpu().numpy()
                    sample_prob = probabilities[i].cpu().numpy()
                    
                    # 计算整体指标
                    overall_metrics = self.calculate_advanced_metrics(sample_pred, sample_target)
                    
                    # 计算每个类别的指标
                    class_metrics = {}
                    for class_idx in range(self.config.num_classes):
                        class_metrics[f'class_{class_idx}'] = self.calculate_advanced_metrics(
                            sample_pred, sample_target, class_idx
                        )
                    
                    sample_result = {
                        'sample_idx': batch_idx * self.config.batch_size + i,
                        'inference_time': inference_time / images.size(0),
                        'overall_metrics': overall_metrics,
                        'class_metrics': class_metrics,
                        'prediction': sample_pred,
                        'target': sample_target,
                        'probabilities': sample_prob
                    }
                    
                    sample_results.append(sample_result)
        
        return sample_results, all_predictions, all_targets
    
    def save_visualization(self, image, target, prediction, probabilities, sample_idx):
        """保存可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        if image.shape[0] == 3:  # RGB图像
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        else:
            img_np = image.squeeze().numpy()
        
        axes[0, 0].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 真实标签
        axes[0, 1].imshow(target, cmap='tab10', vmin=0, vmax=self.config.num_classes-1)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # 预测结果
        axes[0, 2].imshow(prediction, cmap='tab10', vmin=0, vmax=self.config.num_classes-1)
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # 概率图（每个类别）
        for class_idx in range(min(3, self.config.num_classes)):
            if class_idx < probabilities.shape[0]:
                axes[1, class_idx].imshow(probabilities[class_idx], cmap='hot', vmin=0, vmax=1)
                axes[1, class_idx].set_title(f'{self.class_names[class_idx]} Probability')
                axes[1, class_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / f'sample_{sample_idx:04d}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, sample_results, all_predictions, all_targets):
        """生成综合评估报告"""
        print("Generating comprehensive evaluation report...")
        
        # 计算整体统计
        overall_stats = self._calculate_overall_statistics(sample_results)
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)
        
        # 生成分类报告
        class_report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=self.class_names[:self.config.num_classes],
            output_dict=True
        )
        
        # 保存详细结果
        detailed_results = {
            'overall_statistics': overall_stats,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'sample_results': [
                {
                    'sample_idx': r['sample_idx'],
                    'inference_time': r['inference_time'],
                    'overall_metrics': r['overall_metrics']
                } for r in sample_results
            ]
        }
        
        # 保存JSON报告
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # 打印摘要
        self._print_summary(overall_stats, class_report)
        
        return detailed_results
    
    def _calculate_overall_statistics(self, sample_results):
        """计算整体统计信息"""
        stats = defaultdict(list)
        
        for result in sample_results:
            # 整体指标
            for metric, value in result['overall_metrics'].items():
                if not np.isinf(value) and not np.isnan(value):
                    stats[f'overall_{metric}'].append(value)
            
            # 类别指标
            for class_name, class_metrics in result['class_metrics'].items():
                for metric, value in class_metrics.items():
                    if not np.isinf(value) and not np.isnan(value):
                        stats[f'{class_name}_{metric}'].append(value)
            
            # 推理时间
            stats['inference_time'].append(result['inference_time'])
        
        # 计算统计量
        summary_stats = {}
        for key, values in stats.items():
            if values:
                summary_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return summary_stats
    
    def _print_summary(self, overall_stats, class_report):
        """打印评估摘要"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # 整体性能
        print("\n📊 OVERALL PERFORMANCE:")
        print("-" * 40)
        
        key_metrics = ['overall_dice', 'overall_iou', 'overall_precision', 'overall_recall', 'overall_f1_score']
        for metric in key_metrics:
            if metric in overall_stats:
                stats = overall_stats[metric]
                print(f"{metric.replace('overall_', '').replace('_', ' ').title():15}: "
                      f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        # 推理性能
        if 'inference_time' in overall_stats:
            time_stats = overall_stats['inference_time']
            print(f"\n⏱️  INFERENCE PERFORMANCE:")
            print("-" * 40)
            print(f"Average Time: {time_stats['mean']*1000:.2f} ± {time_stats['std']*1000:.2f} ms")
            print(f"Throughput:   {1/time_stats['mean']:.2f} samples/second")
        
        # 各类别性能
        print(f"\n🎯 CLASS-WISE PERFORMANCE:")
        print("-" * 40)
        for i in range(self.config.num_classes):
            if str(i) in class_report:
                class_stats = class_report[str(i)]
                print(f"{self.class_names[i]:15}: "
                      f"P={class_stats['precision']:.3f} "
                      f"R={class_stats['recall']:.3f} "
                      f"F1={class_stats['f1-score']:.3f} "
                      f"Support={class_stats['support']}")
        
        print("\n" + "="*80)
        print(f"📁 Detailed results saved to: {self.output_dir}")
        print("="*80)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Advanced Model Evaluation')
    
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--test_json', type=str, default='test.json',
                       help='Path to test dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=1024,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of classes')
    parser.add_argument('--max_vis_samples', type=int, default=20,
                       help='Maximum number of samples to visualize')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 解析参数
    config = parse_arguments()
    
    print("🚀 Starting Advanced Model Evaluation")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Model: {config.model_path}")
    print(f"Test Data: {config.test_json}")
    print(f"Output Directory: {config.output_dir}")
    
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 加载模型
    model = evaluator.load_model()
    
    # 评估模型
    sample_results, all_predictions, all_targets = evaluator.evaluate_model(model)
    
    # 生成综合报告
    detailed_results = evaluator.generate_comprehensive_report(
        sample_results, all_predictions, all_targets
    )
    
    print("\n✅ Evaluation completed successfully!")
