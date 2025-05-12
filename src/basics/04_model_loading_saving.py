"""
案例4：模型加载与保存
学习目标：
1. 从 Hugging Face Hub 加载模型和 Tokenizer
   - 使用 AutoModel 和 AutoTokenizer 自动识别和加载模型
   - 理解模型和 Tokenizer 的基本使用方法
2. 从本地路径加载模型和 Tokenizer
   - 学习如何将模型保存到本地
   - 掌握从本地加载模型的方法
3. 保存模型和 Tokenizer 到本地
   - 了解模型文件的保存结构
   - 掌握模型配置的保存和加载
4. 理解模型权重文件和配置文件
   - 探索模型文件结构
   - 了解配置文件中的重要参数
"""

import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import BertModel, BertTokenizer, BertConfig

def load_from_hub():
    """
    从 Hugging Face Hub 加载模型和 Tokenizer
    
    功能说明：
    1. 使用 AutoModel 和 AutoTokenizer 自动加载预训练模型
    2. 测试模型的基本功能
    3. 展示模型输出的基本结构
    
    返回：
        tuple: (model, tokenizer) 加载的模型和分词器对象
    """
    print("\n1. 从 Hugging Face Hub 加载模型和 Tokenizer")
    
    # 加载模型和 Tokenizer
    # bert-base-chinese 是一个预训练的中文 BERT 模型
    # 包含约 110M 参数，适合中文 NLP 任务
    model_name = "bert-base-chinese"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"模型类型: {type(model)}")
    print(f"Tokenizer 类型: {type(tokenizer)}")
    
    # 测试模型和 Tokenizer
    # 1. 使用 tokenizer 将文本转换为模型输入格式
    # 2. 将输入传入模型获取输出
    # 3. 查看输出的形状，了解模型的基本结构
    text = "这是一个测试句子。"
    inputs = tokenizer(text, return_tensors="pt")  # 返回 PyTorch 张量
    outputs = model(**inputs)  # 解包输入参数
    
    print(f"输入文本: {text}")
    print(f"模型输出形状: {outputs.last_hidden_state.shape}")
    
    return model, tokenizer

def load_from_local():
    """
    从本地路径加载模型和 Tokenizer
    
    功能说明：
    1. 首先从 Hub 下载模型并保存到本地
    2. 然后从本地加载保存的模型
    3. 验证本地加载的模型功能
    
    返回：
        tuple: (local_model, local_tokenizer) 从本地加载的模型和分词器对象
    """
    print("\n2. 从本地路径加载模型和 Tokenizer")
    
    # 设置保存路径
    model_name = "bert-base-chinese"
    save_path = "./saved_model"
    
    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 加载并保存模型和 Tokenizer
    # 1. 从 Hub 加载模型和 Tokenizer
    # 2. 使用 save_pretrained 保存到本地
    # 3. 保存的文件包括：
    #    - pytorch_model.bin: 模型权重
    #    - config.json: 模型配置
    #    - tokenizer.json: 分词器配置
    #    - vocab.txt: 词表文件
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"模型和 Tokenizer 已保存到: {save_path}")
    
    # 从本地加载
    # 使用相同的 save_path 加载模型和 Tokenizer
    local_model = AutoModel.from_pretrained(save_path)
    local_tokenizer = AutoTokenizer.from_pretrained(save_path)
    
    print(f"本地模型类型: {type(local_model)}")
    print(f"本地 Tokenizer 类型: {type(local_tokenizer)}")
    
    return local_model, local_tokenizer

def explore_model_files():
    """
    探索模型文件结构
    
    功能说明：
    1. 列出保存目录中的所有文件及其大小
    2. 加载并展示模型配置信息
    3. 帮助理解模型文件结构
    """
    print("\n3. 探索模型文件结构")
    
    save_path = "./saved_model"
    
    # 列出保存目录中的文件
    # 显示每个文件的大小（MB）
    print("保存目录中的文件:")
    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为 MB
        print(f"- {file} ({file_size:.2f} MB)")
    
    # 查看配置文件内容
    # 配置文件包含模型的重要参数
    config = AutoConfig.from_pretrained(save_path)
    print("\n模型配置信息:")
    print(f"- 模型类型: {config.model_type}")  # 模型架构类型
    print(f"- 隐藏层大小: {config.hidden_size}")  # 隐藏层维度
    print(f"- 注意力头数量: {config.num_attention_heads}")  # 注意力头数
    print(f"- 层数: {config.num_hidden_layers}")  # Transformer 层数

def main():
    """
    主函数
    
    功能说明：
    1. 按顺序执行三个主要功能：
       - 从 Hub 加载模型
       - 保存并从本地加载模型
       - 探索模型文件结构
    2. 展示完整的模型加载和保存流程
    """
    print("=" * 50)
    print("模型加载与保存示例")
    print("=" * 50)
    
    # 1. 从 Hub 加载
    model, tokenizer = load_from_hub()
    
    # 2. 从本地加载
    local_model, local_tokenizer = load_from_local()
    
    # 3. 探索模型文件
    explore_model_files()

if __name__ == "__main__":
    main() 