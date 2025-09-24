from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class LLM:
    def __init__(self, model_path):
        """
        初始化大语言模型
        
        Args:
            model_path (str): 本地模型文件路径
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        加载本地大语言模型
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                device_map="auto", 
                torch_dtype=torch.float16,

            )
            # Remove self.model.to(self.device) as it conflicts with device_map="auto"
            self.model.eval()
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise e
    
    def response(self, prompt, max_new_tokens=200, temperature=0.7):
        """
        获取模型的输出响应
        
        Args:
            prompt (str): 输入的提示词
            max_new_tokens (int): 最大新生成的token数
            temperature (float): 控制生成文本的随机性
            
        Returns:
            str: 模型生成的响应
        """
        try:
            # 对输入进行编码 - don't manually move to device when using device_map="auto"
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            # The model will automatically handle device placement
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"生成响应时出错: {e}")
            raise e
    
    def extract_drugs_from_prescription(self, prescription_text: str) -> list:
        """从处方文本中提取药物名称"""
        # 移除<output>标签
        prescription_text = re.sub(r'<output>.*?</output>', '', prescription_text, flags=re.DOTALL)
        prescription_text = re.sub(r'</?output>', '', prescription_text)
        
        # 提取中药处方部分
        match = re.search(r'中药处方[：:](.*?)(?:\n|$)', prescription_text)
        if not match:
            return []
        
        drugs_text = match.group(1).strip()
        
        # 分割药物，处理各种分隔符
        drugs = re.split(r'[、，,；;。]', drugs_text)
        
        # 清理药物名称
        cleaned_drugs = []
        for drug in drugs:
            drug = drug.strip()
            # 移除括号内容（如别名）
            drug = re.sub(r'[（(].*?[）)]', '', drug)
            # 移除剂量信息
            drug = re.sub(r'\d+[克g毫ml]', '', drug)
            drug = drug.strip()
            if drug and len(drug) > 1:  # 过滤掉单字符或空字符串
                cleaned_drugs.append(drug)
        
        return list(set(cleaned_drugs))  # 去重

    def with_structured_output(self, schema):
        """
        增强LLM以支持结构化输出
        
        Args:
            schema: 输出结构的模式定义（例如Route类）
            
        Returns:
            一个可以处理结构化输出的对象
        """
        # 创建一个包装器类来处理结构化输出
        class StructuredOutputHandler:
            def __init__(self, llm, schema):
                self.llm = llm
                self.schema = schema
                
            def invoke(self, messages):
                # 将消息转换为提示词
                system_message = ""
                human_message = ""
                
                for message in messages:
                    if hasattr(message, 'content') and hasattr(message, 'type'):
                        if message.type == "system":
                            system_message = message.content
                        elif message.type == "human":
                            human_message = message.content
                
                # 构建用于结构化输出的提示词
                prompt = f"{system_message}\n\n{human_message}\n\n请以JSON格式输出，符合以下模式：{self.schema.__name__}\n"
                prompt += "输出格式示例：{\"step\": \"开药\"} 或 {\"step\": \"知识问答\"} 或 {\"step\": \"相似病案检索\"}\n输出："
                
                # 获取模型响应
                response = self.llm.response(prompt, max_new_tokens=100, temperature=0.1)
                
                # 提取JSON部分
                import json
                import re
                
                # 尝试从响应中提取JSON
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    try:
                        result_dict = json.loads(json_match.group())
                        # 创建schema实例
                        result = self.schema()
                        if 'step' in result_dict:
                            result.step = result_dict['step']
                        return result
                    except json.JSONDecodeError:
                        pass
                
                # 如果无法解析JSON，尝试直接匹配关键词
                response_lower = response.lower()
                if '"step"' in response_lower:
                    if '开药' in response_lower or '开药' in response:
                        result = self.schema()
                        result.step = '开药'
                        return result
                    elif '知识问答' in response_lower or '知识问答' in response:
                        result = self.schema()
                        result.step = '知识问答'
                        return result
                    elif '相似病案' in response_lower or '相似病案' in response:
                        result = self.schema()
                        result.step = '相似病案检索'
                        return result
                
                # 默认返回空的schema实例
                return self.schema()
        
        return StructuredOutputHandler(self, schema)
