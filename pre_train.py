import torch
import argparse
import os
import torch.optim as optim
from model.CellApop import CellApop
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader
from utils.pre_train_dataset import UnLabeled_CellDataset
from torch import nn
from torch.nn import functional as F
from utils.train_loss import mask_loss

def parse_opts():
    parser = argparse.ArgumentParser(description='Pre-training for Student Model')
    parser.add_argument('--pre_train_data_json', default='json', type=str, help='json file path')
    parser.add_argument('--epochs', default=500, type=int, help='Number of total epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers')
    parser.add_argument('--warmup', type=bool, default=True, help='If activated, warm up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=150, help='Warm up iterations, only valid when warmup is activated')
    parser.add_argument('--sam_type', default='vit_b', type=str, help='sam type')
    parser.add_argument('--sam_checkpoint', default='checkpoint/sam/sam_vit_b_01ec64.pth', type=str, help='sam checkpoint')
    parser.add_argument('--model_type', default='cellapop', type=str, help='Student Model type')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--save_dir', default='checkpoint', type=str, help='Model saving directory')
    parser.add_argument('--save_interval', default=5, type=int, help='Model saving interval (epochs)')

    args = parser.parse_args()
    
    return args

def init_sam(model_type, checkpoint):
    # Load model
    model_type = model_type  # or "vit_l", "vit_b"
    checkpoint = checkpoint  # Download corresponding weight file

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return sam

def main():
    args = parse_opts()
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model initialization
    sam = init_sam(args.sam_type, args.sam_checkpoint)
    sam.to(device)
    sam.eval()
    
    model = CellApop(img_size=1024, num_classes=args.num_classes, encoder_fuse=True)
    model.to(device)

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(model.parameters(), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.base_lr / 10)

    # Data loading
    dataset = UnLabeled_CellDataset(args.data_json)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    best_loss = float('inf')
    iter_num = 0
    max_iterations = args.epochs * len(dataloader)

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_embedding_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_dice_loss = 0.0
        
        for i, (ori_img, aug_img, pes_label) in enumerate(dataloader):
            ori_img, aug_img = ori_img.to(device), aug_img.to(device)
            pes_label = pes_label.to(device).float()
            
            with torch.no_grad():
                sam_embedding = sam.image_encoder(aug_img)
            
            mask_pred, cellapop_embedding = model(ori_img)
            mask_pred = torch.sigmoid(mask_pred)

            # Calculate Embedding loss
            embedding_loss = F.mse_loss(cellapop_embedding, sam_embedding)
            
            # Calculate Mask loss
            total_mask_loss, bce_loss, dice_loss_val = mask_loss(
                mask_pred.squeeze(1),
                pes_label,
                alpha=0.7,
                beta=0.3,
                epsilon=0.2
            )
            
            # Total loss
            total_loss = embedding_loss + total_mask_loss
            
            # Backward propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_loss += total_loss.item()
            epoch_embedding_loss += embedding_loss.item()
            epoch_mask_loss += total_mask_loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_dice_loss += dice_loss_val.item()
        
            if args.warmup and iter_num < args.warmup_period:
                # Warm-up phase, learning rate increases linearly from low to high
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    # Post warm-up phase, learning rate gradually decreases
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    # Learning rate adjustment depends on maximum iterations
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1  # Increment iteration counter

        scheduler.step()

        # Average loss for each epoch
        avg_loss = epoch_loss / len(dataloader)
        avg_embedding_loss = epoch_embedding_loss / len(dataloader)
        avg_mask_loss = epoch_mask_loss / len(dataloader)
        avg_bce_loss = epoch_bce_loss / len(dataloader)
        avg_dice_loss = epoch_dice_loss / len(dataloader)
        
        print(f'Epoch [{epoch+1}/{args.epochs}] Summary: Avg Total Loss: {avg_loss:.4f}, Avg Embedding Loss: {avg_embedding_loss:.4f}, Avg Mask Loss: {avg_mask_loss:.4f}, Avg BCE Loss: {avg_bce_loss:.4f}, Avg Dice Loss: {avg_dice_loss:.4f}')
        
        # Save model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.join(args.save_dir, args.model_type), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_type, f'best_model.pth'))
            print(f'Model saved at {os.path.join(args.save_dir, args.model_type, f"best_model.pth")}')
        
if __name__ == '__main__':
    main()
