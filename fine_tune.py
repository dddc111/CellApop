import torch
import argparse
import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model.CellApop import CellApop
from utils.fine_tune_dataset import Labeled_CellDataset
from utils.train_loss import fine_tune_loss
from utils.metrics import dice_coefficient

def parse_opts():
    parser = argparse.ArgumentParser(description='Fine-tune for Student Model')
    parser.add_argument('--fine_tune_train_data_json', default='json', type=str, help='json file path')
    parser.add_argument('--fine_tune_val_data_json', default='json', type=str, help='json file path')
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers')
    parser.add_argument('--warmup', type=bool, default=True, help='If activated, warm up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=150, help='Warm up iterations, only valid when warmup is activated') 
    parser.add_argument('--gamma', type=float, default=0.6, help='Weight for Cross-Entropy Loss')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Weight for Dice Loss')
    parser.add_argument('--pretrained_path', default='checkpoint/cellapop/best_model.pth', type=str, help='cellapop checkpoint')
    parser.add_argument('--model_type', default='cellapop', type=str, help='Student Model type')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--save_dir', default='checkpoint', type=str, help='Model saving directory')

    args = parser.parse_args()

    return args

def load_pretrained_model(pretrained_path, num_classes=3, freeze_encoder=True):

    pretrained_model = CellApop(img_size=1024, num_classes=1, encoder_fuse=True, freeze=freeze_encoder)
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'model' in checkpoint:
        pretrained_model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        pretrained_model.load_state_dict(checkpoint['state_dict'])
    else:
        pretrained_model.load_state_dict(checkpoint)
    
    print("Successfully loaded pretrained weights")
    
    new_model = CellApop(img_size=1024, num_classes=num_classes, encoder_fuse=True, freeze=freeze_encoder)

    pretrained_dict = pretrained_model.state_dict()
    new_model_dict = new_model.state_dict()

    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in new_model_dict and 'final_upsample.3' not in k}

    new_model_dict.update(filtered_dict)
    new_model.load_state_dict(new_model_dict)
    
    print(f"Model transfer completed, output classes: {num_classes}")
    
    return new_model

def main():
    args = parse_opts()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练模型
    model = load_pretrained_model(args.pretrained_path, num_classes=args.num_classes)
    model.to(device)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(model.parameters(), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.base_lr / 10)

    # Data loading
    train_dataset = Labeled_CellDataset(args.fine_tune_train_data_json, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = Labeled_CellDataset(args.fine_tune_val_data_json, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_dice = 0.0
    iter_num = 0
    max_iterations = args.epochs * len(train_dataloader)

    for epoch in range(args.epochs):
        model.train()
        
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs, _ = model(images)  # outputs shape: (N, C, H, W)
            
            # Calculate fine-tune loss (CE + Dice)
            total_loss, ce_loss, dice_loss_val = fine_tune_loss(
                outputs, masks, 
                gamma=args.gamma, 
                epsilon=args.epsilon
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_dice_loss += dice_loss_val.item()
            
            # Warm-up learning rate adjustment
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            
            iter_num += 1
        
        # Learning rate scheduler step
        scheduler.step()
        
        # Calculate average losses for the epoch
        avg_loss = epoch_loss / len(train_dataloader)
        avg_ce_loss = epoch_ce_loss / len(train_dataloader)
        avg_dice_loss = epoch_dice_loss / len(train_dataloader)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] Summary: Avg Total Loss: {avg_loss:.4f}, Avg CE Loss: {avg_ce_loss:.4f}, Avg Dice Loss: {avg_dice_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)

                outputs, _ = model(images)
                loss, _, dice_loss_val = fine_tune_loss(outputs, masks, gamma=args.gamma, epsilon=args.epsilon)
                epoch_val_loss += loss.item()
                epoch_val_dice += dice_coefficient(outputs, masks)
                
            avg_val_loss = epoch_val_loss / len(val_dataloader)
            avg_val_dice = epoch_val_dice / len(val_dataloader)
            print(f'Epoch [{epoch+1}/{args.epochs}] Validation: Avg Loss: {avg_val_loss:.4f}, Avg Dice: {avg_val_dice:.4f}')
                
            if avg_val_dice > best_dice:  
                best_dice = avg_val_dice
                save_path = os.path.join(args.save_dir, args.model_type, 'fine_tune_best_model.pth')
                os.makedirs(os.path.join(args.save_dir, args.model_type), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f'Model saved to {save_path}')
        
if __name__ == '__main__':
    main()