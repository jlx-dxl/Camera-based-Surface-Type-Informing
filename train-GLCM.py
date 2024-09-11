#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
import wandb

from dataset import VisionInformDataset
from model import VisionInform50
from util import *

# Define the device, prioritize using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Camera based friction coefficient estimation using deep learning')
    
    parser.add_argument('--num_workers', type=int, default=24, help="The number of workers to use for data loading")
    parser.add_argument('--dropout_p', type=float, default=0.1, help="The dropout probability to use")   
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--max_epoch', type=int, default=100, help="Maximum number of epochs")
    parser.add_argument('--N', type=int, default=14, help="Number of classes")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.9, help="Learning rate decay rate")
    parser.add_argument('--patience', type=int, default=20, help="Patience for learning rate scheduler")
    parser.add_argument('--experiment_name', type=str, default='testing', help='The name of the experiment to run')
    parser.add_argument('--use_wandb', action='store_true', default=False, help="If set, use wandb to keep track of experiments")
    parser.add_argument('--wandb-project', type=str, default='Vision-Inform-Friction-Coefficient-Estimation',help='Weights and Biases project name')
    parser.add_argument('--continue_training', action='store_true', default=False, help='If set, continue training from the last best checkpoint')
    
    args = parser.parse_args()
    return args

def train_one_epoch(args, model, optimizer, criterion, train_dataloader):
    """
    Trains the model for one epoch.

    Args:
        args (argparse.Namespace): Parsed arguments.
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        train_dataloader (DataLoader): DataLoader for training data.

    Returns:
        tuple: Average training loss and accuracy.
    """
    model.train()
    total_loss = 0
    total_acc = 0

    for inputs, GLCMs, labels, class_gt in tqdm(train_dataloader):
        inputs = inputs.to(device)
        GLCMs = GLCMs.to(device)
        labels = labels.to(device)
        class_gt = class_gt.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, GLCMs)
        print("outputs: ", outputs.size())
        outputs = outputs.squeeze(1)
        
        class_estimation = get_class_name(outputs).to(device)
        acc = torch.sum(class_gt == class_estimation).item() / class_gt.size(0)
        total_acc += acc
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input.size(0)
        
        if args.use_wandb:
            wandb.log({"loss": loss, "acc": acc, "lr": optimizer.param_groups[0]['lr']})
            
    return total_loss / len(train_dataloader.dataset), total_acc / train_dataloader.__len__()

def evaluate_one_epoch(model, criterion, dev_dataloader):
    """
    Evaluates the model for one epoch.

    Args:
        model (nn.Module): The model to be evaluated.
        criterion (nn.Module): Loss function.
        dev_dataloader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for inputs, labels, class_gt in tqdm(dev_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_gt = class_gt.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            
            class_estimation = get_class_name(outputs).to(device)
            acc = torch.sum(class_gt == class_estimation).item() / class_gt.size(0)
            total_acc += acc
        
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input.size(0)
        
    return total_loss / len(dev_dataloader.dataset), total_acc / dev_dataloader.__len__()

def main():
    """
    Main function to train and evaluate the model.
    """
    args = get_args()
    
    if args.use_wandb:
        setup_wandb(args, args.experiment_name)
        
    dropout_p = args.dropout_p
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    N = args.N
    lr = args.lr
    lr_decay = args.lr_decay
    patience = args.patience
    experiment_name = args.experiment_name
    num_workers = args.num_workers
    continue_training = args.continue_training
    
    checkpoint_dir = os.path.join(f'model/{experiment_name}/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create DataLoader
    train_dataloader = DataLoader(VisionInformDataset(set_type='train', if_GLCM=False), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(VisionInformDataset(set_type='test', if_GLCM=False), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    model = VisionInform50(dropout_p=dropout_p, N=N).to(device)
    
    print(f"Model is training on {next(model.parameters()).device} !!!")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=20)
    criterion = nn.CrossEntropyLoss()
    
    if continue_training:
        model.load_state_dict(torch.load(os.path.join(f'model/{experiment_name}/best.pth')))
        print("Continue training from the last best checkpoint!!!")
    
    best_loss = float('inf')
    best_acc = 0
    
    for epoch in range(max_epoch):
        train_loss, train_acc = train_one_epoch(args, model, optimizer, criterion, train_dataloader)
        test_loss, test_acc = evaluate_one_epoch(model, criterion, test_dataloader)
        
        print(f"Epoch {epoch+1}/{max_epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Dev Loss: {test_loss:.4f}")
        print(f"Train Acc: {train_acc * 100:.4f}%, Dev Acc: {test_acc * 100:.4f}%")
        
        scheduler.step()
        
        # Save checkpoint
        if test_acc < best_loss and test_acc > best_acc:
            best_loss = test_loss
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,'best.pth'))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f'{epoch+1}.pth'))
            print(f'Checkpoint saved at Epoch {epoch+1}')
        
        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "dev_loss": test_loss, "train_acc": train_acc, "dev_acc": test_acc, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch+1})
            
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
