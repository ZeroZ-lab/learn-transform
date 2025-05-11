"""
案例1：Transformers 架构基础与第一个模型推理

本案例介绍 Hugging Face Transformers 的基本架构，并通过 pipeline API 实现第一个情感分析推理。

学习目标：
- 理解 Transformers 库的核心组成
- 了解 pipeline 的用法
- 体验零代码加载和推理预训练模型

相关文档：
https://huggingface.co/docs/transformers/index

企业级最佳实践：
- 代码结构清晰，注释详细
- 只用官方 API，避免自造轮子
- 便于后续扩展和测试
"""

from transformers import pipeline

def main():
    # 创建情感分析 pipeline
    classifier = pipeline("sentiment-analysis")
    
    # 测试文本
    texts = [
        "I love using Hugging Face Transformers!",
        "This is a terrible experience.",
    ]
    
    # 推理
    results = classifier(texts)
    for text, result in zip(texts, results):
        print(f"文本: {text}\n结果: {result}\n")

if __name__ == "__main__":
    main()
