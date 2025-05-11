# Learn Transformers

这个项目旨在通过实践来学习 Hugging Face Transformers 库。我们将通过一系列循序渐进的示例来掌握 Transformers 的核心概念和应用。

## 项目结构

```
learn-transform/
├── src/                    # 源代码目录
│   ├── basics/            # 基础概念和工具
│   ├── models/            # 模型相关实现
│   ├── datasets/          # 数据集处理
│   ├── training/          # 训练相关代码
│   └── advanced/          # 高级应用
├── docs/                  # 文档
├── examples/              # 示例代码
├── tests/                 # 测试代码
└── requirements.txt       # 项目依赖
```

## 学习路线

### 1. 基础知识 (src/basics)
- [案例1：Transformers 架构基础与第一个模型推理](src/basics/01_transformer_architecture.py)
  - 介绍 Transformers 架构，使用 pipeline API 实现情感分析推理
- [案例2：Tokenization 原理与实践](src/basics/02_tokenization.py)
  - 学习文本分词、编码、解码等基础操作，掌握中英文处理
- 预训练模型基础概念
- 模型加载与保存
- 配置管理

### 2. 模型应用 (src/models)
- 文本分类
- 命名实体识别 (NER)
- 问答系统
- 文本生成
- 机器翻译
- 情感分析

### 3. 数据处理 (src/datasets)
- 数据集加载与处理
- 数据预处理
- 数据增强
- 批处理与数据加载器
- 自定义数据集

### 4. 模型训练 (src/training)
- 模型微调
- 训练配置
- 优化器选择
- 学习率调度
- 训练监控与评估
- 分布式训练

### 5. 高级应用 (src/advanced)
- 模型量化
- 模型蒸馏
- 多任务学习
- 跨语言迁移
- 模型部署
- 性能优化

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 其他依赖见 requirements.txt

## 安装

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

## 使用说明

每个示例都包含详细的注释和说明文档，建议按照以下顺序学习：

1. 从 basics 目录开始，掌握基础概念
2. 通过 examples 目录中的示例进行实践
3. 参考 docs 目录中的文档深入了解
4. 尝试修改和扩展示例代码

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个学习项目。

## 许可证

MIT License
