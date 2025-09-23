import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class TCMPrescriptionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_herbs=301, hidden_dim=768, dropout=0.1):
        """
        中药处方生成模型
        
        Args:
            bert_model_name: BERT模型名称
            num_herbs: 药材种类数量
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(TCMPrescriptionModel, self).__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 多标签分类头部
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_herbs)
        
        # 可选的额外层
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            
        Returns:
            logits: 预测logits
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用CLS token表示整个序列
        pooled_output = outputs.pooler_output
        
        # 应用dropout
        pooled_output = self.dropout(pooled_output)
        
        # 可选的隐藏层
        # hidden = self.hidden_activation(self.hidden_layer(pooled_output))
        # hidden = self.dropout(hidden)
        
        # 分类层
        logits = self.classifier(pooled_output)
        
        return logits

class AdvancedTCMPrescriptionModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_herbs=301, hidden_dim=768, dropout=0.1):
        """
        改进版中药处方生成模型
        
        Args:
            bert_model_name: BERT模型名称
            num_herbs: 药材种类数量
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(AdvancedTCMPrescriptionModel, self).__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # 多层分类头部
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_herbs)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            
        Returns:
            logits: 预测logits
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # 应用自注意力机制
        attended_output, _ = self.attention(sequence_output, sequence_output, sequence_output, 
                                          key_padding_mask=~attention_mask.bool())
        
        # 残差连接和层归一化
        attended_output = self.layer_norm(sequence_output + attended_output)
        
        # 使用CLS token表示整个序列
        pooled_output = attended_output[:, 0, :]  # [batch_size, hidden_dim]
        
        # 应用dropout
        pooled_output = self.dropout(pooled_output)
        
        # 多层感知机
        hidden = self.relu(self.fc1(pooled_output))
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        
        # 分类层
        logits = self.classifier(hidden)
        
        return logits