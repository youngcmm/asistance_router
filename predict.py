import torch
from transformers import BertTokenizer
import json
import re
import numpy as np
import os
from model import TCMPrescriptionModel
from data_preprocessing import TCMDataPreprocessor
from evaluate_herbs.evaluate_herbs import evaluate_herbs
from llm import LLM

class TCMPredictor:
    def __init__(self, model_path, herb_file_path, model_config=None):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            herb_file_path: 药材文件路径
            model_config: 模型配置
        """
        self.tokenizer_path = '.model/bert_tokenizer'
        self.herb_file_path = herb_file_path
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # 加载本地分词器，如果不存在则下载到本地
        self.tokenizer = self.load_or_download_tokenizer()

        # 加载药材列表
        self.load_herb_list()
        
        # 创建模型
        if model_config is None:
            self.model = TCMPrescriptionModel(num_herbs=len(self.herb_list))
        else:
            self.model = TCMPrescriptionModel(**model_config)
        
        # 加载模型权重（如果模型文件存在）
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Model file not found, using untrained model")
        except Exception as e:
            print(f"Error loading model: {e}, using untrained model")
        
        self.model.eval()
    def load_or_download_tokenizer(self):
        """
        加载本地分词器，如果不存在则从网络下载并保存到本地
        
        Returns:
            tokenizer: BERT分词器
        """
        try:
            # 尝试从本地加载分词器
            if os.path.exists(self.tokenizer_path):
                tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
                print(f"Tokenizer loaded from local path: {self.tokenizer_path}")
            else:
                # 如果本地不存在，则从网络下载
                print("Local tokenizer not found, downloading from HuggingFace...")
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                
                # 保存到本地
                os.makedirs(self.tokenizer_path, exist_ok=True)
                tokenizer.save_pretrained(self.tokenizer_path)
                print(f"Tokenizer downloaded and saved to: {self.tokenizer_path}")
                
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to downloading tokenizer...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 尝试保存到本地（如果路径可用）
            try:
                os.makedirs(self.tokenizer_path, exist_ok=True)
                tokenizer.save_pretrained(self.tokenizer_path)
                print(f"Tokenizer saved to: {self.tokenizer_path}")
            except Exception as save_error:
                print(f"Could not save tokenizer locally: {save_error}")
                
            return tokenizer

    def load_herb_list(self):
        """加载药材列表"""
        self.herb_list = []
        try:
            # 首先尝试加载 herb_list.txt (清洗后的文件)
            with open(self.herb_file_path.replace('herb_count.txt', 'herb_list.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    herb = line.strip()
                    if herb:
                        self.herb_list.append(herb)
        except FileNotFoundError:
            # 如果 herb_list.txt 不存在，则从 herb_count.txt 加载
            with open(self.herb_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    herb = line.split(':')[0].strip()
                    self.herb_list.append(herb)
        
        self.herb_to_idx = {herb: idx for idx, herb in enumerate(self.herb_list)}
        self.idx_to_herb = {idx: herb for idx, herb in enumerate(self.herb_list)}
        print(f"Loaded {len(self.herb_list)} herbs")
        
    def preprocess_input(self, patient_info, max_length=512):
        """
        预处理输入文本
        
        Args:
            patient_info: 患者信息文本
            max_length: 最大序列长度
            
        Returns:
            encoded: 编码后的输入
        """
        encoded = self.tokenizer.encode_plus(
            patient_info,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def predict(self, patient_info, threshold=0.5, top_k=None):
        """
        预测处方
        
        Args:
            patient_info: 患者信息
            threshold: 阈值
            top_k: 返回前k个预测结果
            
        Returns:
            predicted_herbs: 预测的药材列表
            probabilities: 对应概率
        """
        # 预处理输入
        inputs = self.preprocess_input(patient_info)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # 应用sigmoid获取概率
            probabilities = torch.sigmoid(outputs).squeeze().numpy()
        
        # 根据阈值或top_k选择药材
        if top_k is not None:
            # 选择概率最高的top_k个
            top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(probabilities[top_indices])[::-1]]  # 按概率排序
            predicted_herbs = [self.idx_to_herb[idx] for idx in top_indices]
            selected_probs = [probabilities[idx] for idx in top_indices]
        else:
            # 根据阈值选择
            predicted_indices = np.where(probabilities > threshold)[0]
            predicted_herbs = [self.idx_to_herb[idx] for idx in predicted_indices]
            selected_probs = [probabilities[idx] for idx in predicted_indices]
            
            # 按概率排序
            sorted_indices = np.argsort(selected_probs)[::-1]
            predicted_herbs = [predicted_herbs[i] for i in sorted_indices]
            selected_probs = [selected_probs[i] for i in sorted_indices]
        
        return predicted_herbs, selected_probs
    
    def format_prescription(self, herbs):
        """
        格式化处方
        
        Args:
            herbs: 药材列表
            
        Returns:
            formatted: 格式化的处方文本
        """
        return "中药处方：" + "、".join(herbs)
    
    def load_test_data(self, data_path='/media/data4/yangcm/multi_class/data/processed_data_test.json'):
        """
        加载测试数据
        
        Args:
            data_path: 数据路径
            
        Returns:
            test_cases: 测试案例列表
            grts:  Ground Truth列表
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        return test_data
def main():
    # 初始化预测器
    predictor = TCMPredictor(
        model_path='best_model.pth',
        herb_file_path='data/herb_count.txt'
    )
    func_list = []
    with open('/media/data4/yangcm/test_local_model/evaluate_herbs/herb2func.json', 'r', encoding='utf-8') as f:
            func_list = json.load(f)
    f.close()
    
    test_data = predictor.load_test_data(data_path='/media/data4/yangcm/LLaMA-Factory/data/processed_100_data_test.json')
    input_list, grt_list = [], []
    for data in test_data:
        input_list.append(data['input'])
        grt_list.append(data['output'][5:])
    # input_list = [input_item for input_item in test_data['inputs']]
    # inpput_list = test_data['inputs']
    # grt_list = [grt_item for grt_item[5:] in test_data['output']]
    # print(input_list)
    # print(grt_list)
    # 示例预测
    # test_cases = [
    #     "年龄：43岁，现病史：间有失眠，舌苔脉象：舌齿印 苔薄白 脉弦偏弱"
    # ]
    # grt = [
    #     '川芎、党参、白术、当归、茯苓、白芍、甘草、地黄'
    # ]
    precision_value_list = []
    recall_value_list = []
    f1_value_list = []
    # llm = LLM(model_path = "/media/data4/yangcm/LLaMA-Factory/output/qwen2_5_7b_lora_sft_6000_merged")
    llm = LLM(model_path = "/media/data2/yangcm/models/Qwen2.5-14B-Instruct")
    
    for i, case in enumerate(input_list):
        try:
            print(f"\n测试案例 {i+1}:")
            print(f"患者信息: {case}")
            
            # 预测
            herbs, probs = predictor.predict(case, top_k=15)
            
            # pred = '茯苓、酸枣仁、甘草、柴胡、白术、首乌藤、合欢皮、党参、龙骨、牡蛎、白芍、远志、陈皮、郁金、当归'
            # grt = '川芎、甘草、知母、丹参、川芎、当归、酸枣仁、黄芪、白术、甘草、知母、茯苓、酸枣仁、黄芪、茯苓、白术、丹参、当归'
            # print(func_list)
            
            pred = herbs
            grt = grt_list[i]
            
            ######### llm
            prompt = f"患者信息: {case}\n\n可能会用到的药{herbs}。根据上述信息生成中药处方。\n\n严格按照如下的格式输出\n\n中药处方：药材名称、药材名称..."
            res = llm.response(prompt=prompt)
            # print(res)
            pred = llm.extract_drugs_from_prescription(res)
            
            precision, recall, f1 = evaluate_herbs(pred, grt, func_list, 1, 'f1', False)
        except:
            continue    
        if precision == 0 or recall == 0 or f1 == 0:
            continue
        precision_value_list.append(precision)
        recall_value_list.append(recall)
        f1_value_list.append(f1)

        # print('precision: ', precision, 'recall: ', recall, 'f1: ', f1)

        # 格式化处方
        # prescription = predictor.format_prescription(herbs)
        # print(f"推荐处方: {prescription}")
        # print("各药材概率:")
        # for herb, prob in zip(herbs[:10], probs[:10]):  # 只显示前10个
            # print(f"  {herb}: {prob:.4f}")

    print('precision: ', np.mean(precision_value_list), end=' ')
    print(np.std(precision_value_list))

    print('recall: ', np.mean(recall_value_list), end=' ')
    print(np.std(recall_value_list))

    print('f1: ', np.mean(f1_value_list), end=' ')
    print(np.std(f1_value_list))

if __name__ == "__main__":
    main()