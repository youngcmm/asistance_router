import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
import numpy as np
import json
from tqdm import tqdm
import os

from data_preprocessing import TCMDataPreprocessor, TCMDataset
from model import TCMPrescriptionModel, AdvancedTCMPrescriptionModel

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_f1 = 0.0
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            try:
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
                
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                try:
                    # 将数据移到设备上
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 应用sigmoid获取概率并转换为二进制预测
                    probs = torch.sigmoid(outputs)
                    predictions = (probs > 0.5).float()
                    
                    # 转换为numpy数组用于计算指标
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        if len(all_predictions) == 0:
            return {
                'loss': 0,
                'hamming_loss': 0,
                'f1_macro': 0,
                'f1_micro': 0,
                'precision': 0,
                'recall': 0
            }
        
        # 计算评估指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        hamming = hamming_loss(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'hamming_loss': hamming,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, num_epochs, save_path='best_model.pth'):
        """完整训练流程"""
        train_losses = []
        val_losses = []
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # 评估
            metrics = self.evaluate()
            val_losses.append(metrics['loss'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {metrics['loss']:.4f}")
            print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"F1 Micro: {metrics['f1_micro']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
            # 保存最佳模型
            if metrics['f1_macro'] > best_f1:
                best_f1 = metrics['f1_macro']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_macro': best_f1,
                }, save_path)
                print(f"New best model saved with F1 Macro: {best_f1:.4f}")
        
        return train_losses, val_losses

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    preprocessor = TCMDataPreprocessor(
        herb_file_path='data/herb_count.txt',
        train_file_path='data/processed_data_train.json',
        test_file_path='data/processed_data_test.json'
    )
    
    (train_encodings, train_labels), (test_encodings, test_labels) = preprocessor.preprocess()
    
    print(f"Train data shape: {train_labels.shape}")
    print(f"Test data shape: {test_labels.shape}")
    
    # 创建数据集和数据加载器
    train_dataset = TCMDataset(train_encodings, train_labels)
    test_dataset = TCMDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = TCMPrescriptionModel(num_herbs=len(preprocessor.herb_list))
    model.to(device)
    
    # 定义损失函数和优化器
    # 使用带正负样本均衡的BCEWithLogitsLoss
    pos_weight = torch.tensor([1.0] * len(preprocessor.herb_list)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device)
    
    # 开始训练
    train_losses, val_losses = trainer.train(num_epochs=3)
    
    print("Training completed!")

if __name__ == "__main__":
    main()