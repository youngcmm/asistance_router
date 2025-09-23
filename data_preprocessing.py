import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import jieba
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

class TCMDataPreprocessor:
    def __init__(self, herb_file_path, train_file_path, test_file_path):
        self.herb_file_path = herb_file_path
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.herb_list = []
        self.herb_to_idx = {}
        self.idx_to_herb = {}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
    def load_herb_list(self):
        """加载药材列表"""
        with open(self.herb_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                herb = line.split(':')[0].strip()
                self.herb_list.append(herb)
        
        # 创建索引映射
        self.herb_to_idx = {herb: idx for idx, herb in enumerate(self.herb_list)}
        self.idx_to_herb = {idx: herb for idx, herb in enumerate(self.herb_list)}
        print(f"Loaded {len(self.herb_list)} herbs")
        
    def extract_herbs_from_prescription(self, prescription_text):
        """从处方文本中提取药材列表"""
        # 移除"中药处方："前缀
        if "：" in prescription_text:
            prescription_text = prescription_text.split("：", 1)[1]
        
        # 按顿号、逗号分割
        herbs = re.split('[、,，]', prescription_text.strip())
        # 清理每个药材名称
        herbs = [herb.strip() for herb in herbs if herb.strip()]
        return herbs
    
    def load_data(self, file_path):
        """加载训练或测试数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        inputs = []
        labels = []
        
        for item in data:
            inputs.append(item['input'])
            # 提取处方中的药材
            herbs = self.extract_herbs_from_prescription(item['output'])
            labels.append(herbs)
            
        return inputs, labels
    
    def encode_texts(self, texts, max_length=512):
        """编码文本数据"""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].flatten())
            attention_masks.append(encoded['attention_mask'].flatten())
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }
    
    def create_label_matrix(self, labels):
        """创建多标签矩阵"""
        label_matrix = np.zeros((len(labels), len(self.herb_list)))
        
        for i, herb_list in enumerate(labels):
            for herb in herb_list:
                if herb in self.herb_to_idx:
                    label_matrix[i][self.herb_to_idx[herb]] = 1
                    
        return label_matrix
    
    def preprocess(self):
        """预处理所有数据"""
        # 加载药材列表
        self.load_herb_list()
        
        # 加载训练和测试数据
        train_inputs, train_labels = self.load_data(self.train_file_path)
        test_inputs, test_labels = self.load_data(self.test_file_path)
        
        # 编码文本
        train_encoded = self.encode_texts(train_inputs)
        test_encoded = self.encode_texts(test_inputs)
        
        # 创建标签矩阵
        train_label_matrix = self.create_label_matrix(train_labels)
        test_label_matrix = self.create_label_matrix(test_labels)
        
        return (train_encoded, train_label_matrix), (test_encoded, test_label_matrix)

class TCMDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)