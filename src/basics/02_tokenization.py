"""
案例2：Tokenization 原理与实践

本案例介绍 Transformers 中的 Tokenization 概念，包括：
- 什么是 Tokenization
- 常见的 Tokenization 方法
- 如何使用 AutoTokenizer
- 处理中文文本
- 处理特殊字符和标点

【Tokenization 的典型使用场景】
1. 文本分类：如情感分析、垃圾邮件检测，将原始文本转为 token 序列供模型分类。
2. 序列标注：如命名实体识别（NER）、分词，对每个 token 进行标签预测。
3. 问答与阅读理解：将问题和上下文拼接后 tokenization，满足模型输入格式。
4. 文本生成：如机器翻译、摘要生成，将输入文本编码为 token，输出 token 再解码为自然语言。
5. 多语言处理：支持不同语言的分词规则和词表，提升多语言模型泛化能力。
6. 自定义数据集与预处理：批量处理大规模文本，生成模型训练所需的输入格式。
7. 特殊符号与结构化文本处理：如代码生成、公式识别，保证 token 序列的语义完整性。

学习目标：
- 理解 Tokenization 的基本原理
- 掌握 AutoTokenizer 的使用方法
- 了解不同 Tokenizer 的特点
- 学会处理常见的中英文文本

企业级最佳实践：
- 使用 AutoTokenizer 实现跨模型兼容
- 正确处理特殊字符和标点
- 注意 token 长度限制
- 处理多语言文本
"""

from transformers import AutoTokenizer

def basic_tokenization():
    """基础 tokenization 示例"""
    # 使用 BERT 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 简单文本
    text = "我爱使用 Hugging Face Transformers！"
    
    # 进行 tokenization
    tokens = tokenizer.tokenize(text)
    print("分词结果:", tokens)
    
    # 转换为 ID
    ids = tokenizer.encode(text)
    print("Token IDs:", ids)
    
    # 解码回文本
    decoded = tokenizer.decode(ids)
    print("解码结果:", decoded)

def advanced_tokenization():
    """高级 tokenization 示例"""
    # 使用 GPT2 的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 处理特殊字符和标点
    text = "Hello, World! This is a test... (with special characters)"
    
    # 添加特殊 token
    tokens = tokenizer.tokenize(text, add_special_tokens=True)
    print("\n特殊字符处理:", tokens)
    
    # 处理长文本
    long_text = "This is a very long text that needs to be truncated. " * 10
    truncated = tokenizer(long_text, truncation=True, max_length=50)
    print("\n截断处理:", len(truncated["input_ids"]))

def chinese_tokenization():
    """中文 tokenization 示例"""
    # 使用中文 BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 中文文本
    text = "人工智能正在改变世界，Transformers 是其中的重要技术。"
    
    # 分词
    tokens = tokenizer.tokenize(text)
    print("\n中文分词:", tokens)
    
    # 转换为 ID 并解码
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print("中文解码:", decoded)

def main():
    print("=== 基础 Tokenization 示例 ===")
    basic_tokenization()
    
    print("\n=== 高级 Tokenization 示例 ===")
    advanced_tokenization()
    
    print("\n=== 中文 Tokenization 示例 ===")
    chinese_tokenization()

if __name__ == "__main__":
    main() 