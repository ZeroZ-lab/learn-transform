"""
案例1：文本分类
学习目标：
1. 使用 AutoModelForSequenceClassification 进行文本分类
   - 掌握模型加载和初始化
   - 理解输入数据的准备
   - 掌握模型输出的解析
2. 实践微调预训练模型
   - 了解微调的基本流程
   - 掌握数据准备和训练过程
3. 模型评估和预测
   - 学习如何评估模型性能
   - 掌握如何进行预测
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def prepare_model_and_tokenizer():
    """
    准备模型和分词器
    
    功能说明：
    1. 加载预训练模型和分词器
    2. 配置模型用于分类任务
    """
    print("\n1. 准备模型和分词器")
    
    # 加载预训练模型和分词器
    model_name = "bert-base-chinese"
    num_labels = 2  # 二分类任务
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型并配置为分类任务
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    print(f"模型类型: {type(model)}")
    print(f"分类标签数: {num_labels}")
    
    return model, tokenizer

def prepare_sample_data():
    """
    准备示例数据
    
    功能说明：
    1. 创建简单的二分类数据集
    2. 展示数据预处理流程
    """
    print("\n2. 准备示例数据")
    
    # 创建示例数据
    texts = [
        "这部电影很棒，我很喜欢！",
        "这个产品质量太差了，不推荐购买。",
        "服务态度很好，环境也不错。",
        "价格太贵了，不值这个价。",
        "这个餐厅的菜很好吃，下次还会来。",
        "这个产品用起来很方便，推荐购买。",
        "服务太差了，等了很久。",
        "这个价格很合理，性价比高。"
    ]
    
    labels = [1, 0, 1, 0, 1, 1, 0, 1]  # 1: 正面, 0: 负面
    
    # 创建数据集
    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })
    
    print(f"数据集大小: {len(dataset)}")
    print("\n示例数据:")
    for i in range(2):
        print(f"文本: {texts[i]}")
        print(f"标签: {labels[i]}")
    
    return dataset

def preprocess_data(dataset, tokenizer):
    """
    数据预处理
    
    功能说明：
    1. 对文本进行分词和编码
    2. 准备模型输入格式
    """
    print("\n3. 数据预处理")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # 对数据集进行分词
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print("预处理后的数据集特征:")
    print(tokenized_dataset.features)
    
    return tokenized_dataset

def compute_metrics(pred):
    """
    计算评估指标
    
    功能说明：
    1. 计算准确率、精确率、召回率和 F1 分数
    2. 用于模型评估
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model, tokenized_dataset):
    """
    训练模型
    
    功能说明：
    1. 配置训练参数
    2. 使用 Trainer 进行训练
    """
    print("\n4. 训练模型")
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    trainer.train()
    
    return trainer

def predict_sentiment(model, tokenizer, text):
    """
    预测文本情感
    
    功能说明：
    1. 对输入文本进行预测
    2. 返回预测结果和概率
    """
    # 准备输入
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 获取预测结果
    predicted_class = predictions.argmax().item()
    confidence = predictions[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    """
    主函数
    
    功能说明：
    1. 按顺序执行主要功能：
       - 准备模型和分词器
       - 准备示例数据
       - 数据预处理
       - 训练模型
       - 进行预测
    2. 展示完整的文本分类流程
    """
    print("=" * 50)
    print("文本分类示例")
    print("=" * 50)
    
    # 1. 准备模型和分词器
    model, tokenizer = prepare_model_and_tokenizer()
    
    # 2. 准备示例数据
    dataset = prepare_sample_data()
    
    # 3. 数据预处理
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    
    # 4. 训练模型
    trainer = train_model(model, tokenized_dataset)
    
    # 5. 进行预测
    print("\n5. 预测示例")
    test_texts = [
        "这个产品非常好用，推荐购买！",
        "服务态度很差，不推荐。"
    ]
    
    for text in test_texts:
        predicted_class, confidence = predict_sentiment(model, tokenizer, text)
        sentiment = "正面" if predicted_class == 1 else "负面"
        print(f"\n文本: {text}")
        print(f"情感: {sentiment}")
        print(f"置信度: {confidence:.2f}")

if __name__ == "__main__":
    main() 