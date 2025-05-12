"""
案例5：模型配置 Configuration
学习目标：
1. 了解 AutoConfig 和具体模型的 Config 对象
   - 掌握不同模型配置类的使用方法
   - 理解配置对象的作用和重要性
2. 学习查看和修改模型配置参数
   - 了解常用配置参数的含义
   - 掌握如何自定义模型配置
3. 理解配置在模型初始化和行为中的作用
   - 配置如何影响模型结构
   - 配置如何影响模型行为
"""

from transformers import AutoConfig, BertConfig, GPT2Config
from transformers import AutoModel, AutoModelForCausalLM
import torch

def explore_bert_config():
    """
    探索 BERT 模型的配置
    
    功能说明：
    1. 展示如何加载和查看 BERT 配置
    2. 展示如何修改配置参数
    3. 使用修改后的配置初始化模型
    """
    print("\n1. BERT 模型配置示例")
    
    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained("bert-base-chinese")
    print("\n原始配置信息:")
    print(f"- 模型类型: {config.model_type}")
    print(f"- 隐藏层大小: {config.hidden_size}")
    print(f"- 注意力头数量: {config.num_attention_heads}")
    print(f"- 层数: {config.num_hidden_layers}")
    print(f"- 词表大小: {config.vocab_size}")
    
    # 创建新的配置
    custom_config = BertConfig(
        vocab_size=21128,  # 词表大小
        hidden_size=512,   # 隐藏层维度
        num_hidden_layers=6,  # Transformer 层数
        num_attention_heads=8,  # 注意力头数
        intermediate_size=2048,  # 前馈网络维度
        max_position_embeddings=512,  # 最大位置编码长度
        type_vocab_size=2,  # token 类型数量
    )
    
    print("\n自定义配置信息:")
    print(f"- 隐藏层大小: {custom_config.hidden_size}")
    print(f"- 注意力头数量: {custom_config.num_attention_heads}")
    print(f"- 层数: {custom_config.num_hidden_layers}")
    
    # 使用自定义配置初始化模型
    model = AutoModel.from_config(custom_config)
    print(f"\n使用自定义配置创建的模型类型: {type(model)}")
    
    return model, custom_config

def explore_gpt2_config():
    """
    探索 GPT-2 模型的配置
    
    功能说明：
    1. 展示 GPT-2 特有的配置参数
    2. 展示如何修改 GPT-2 的配置
    3. 使用修改后的配置初始化模型
    """
    print("\n2. GPT-2 模型配置示例")
    
    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained("gpt2")
    print("\n原始 GPT-2 配置信息:")
    print(f"- 模型类型: {config.model_type}")
    print(f"- 隐藏层大小: {config.hidden_size}")
    print(f"- 注意力头数量: {config.num_attention_heads}")
    print(f"- 层数: {config.num_hidden_layers}")
    print(f"- 词表大小: {config.vocab_size}")
    
    # 创建新的 GPT-2 配置
    custom_config = GPT2Config(
        vocab_size=50257,  # 词表大小
        n_positions=1024,  # 最大位置编码长度
        n_ctx=1024,        # 上下文窗口大小
        n_embd=768,        # 嵌入维度
        n_layer=6,         # 层数
        n_head=12,         # 注意力头数
    )
    
    print("\n自定义 GPT-2 配置信息:")
    print(f"- 嵌入维度: {custom_config.n_embd}")
    print(f"- 注意力头数量: {custom_config.n_head}")
    print(f"- 层数: {custom_config.n_layer}")
    
    # 使用自定义配置初始化模型
    model = AutoModelForCausalLM.from_config(custom_config)
    print(f"\n使用自定义配置创建的模型类型: {type(model)}")
    
    return model, custom_config

def compare_configs():
    """
    比较不同模型的配置差异
    
    功能说明：
    1. 展示不同模型架构的配置差异
    2. 理解配置参数对模型结构的影响
    """
    print("\n3. 配置比较")
    
    # 加载不同模型的配置
    bert_config = AutoConfig.from_pretrained("bert-base-chinese")
    gpt2_config = AutoConfig.from_pretrained("gpt2")
    
    print("\nBERT vs GPT-2 配置比较:")
    print("BERT 特有参数:")
    print(f"- type_vocab_size: {bert_config.type_vocab_size}")
    print(f"- layer_norm_eps: {bert_config.layer_norm_eps}")
    
    print("\nGPT-2 特有参数:")
    print(f"- n_positions: {gpt2_config.n_positions}")
    print(f"- n_ctx: {gpt2_config.n_ctx}")
    print(f"- scale_attn_weights: {gpt2_config.scale_attn_weights}")

def main():
    """
    主函数
    
    功能说明：
    1. 按顺序执行三个主要功能：
       - 探索 BERT 配置
       - 探索 GPT-2 配置
       - 比较不同模型的配置
    2. 展示配置对象的使用方法
    """
    print("=" * 50)
    print("模型配置示例")
    print("=" * 50)
    
    # 1. 探索 BERT 配置
    bert_model, bert_config = explore_bert_config()
    
    # 2. 探索 GPT-2 配置
    gpt2_model, gpt2_config = explore_gpt2_config()
    
    # 3. 比较配置
    compare_configs()

if __name__ == "__main__":
    main() 