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
- `[案例1：Transformers 架构与 Pipeline API](src/basics/01_transformer_architecture.py)`
  - 理解 Transformer 模型的核心架构（Encoder-Decoder, Self-Attention）
  - 学习使用 `pipeline` API 进行快速模型推理（如情感分析、文本生成）
  - 了解不同任务对应的 Pipeline 类型
- `[案例2：Tokenization 原理与实践](src/basics/02_tokenization.py)`
  - 掌握 Tokenizer 的基本工作流程：文本规范化、预分词、模型分词、ID转换
  - 学习使用 `AutoTokenizer` 加载不同模型的 Tokenizer
  - 实践文本的编码 (`encode`, `__call__`) 与解码 (`decode`)
  - 理解 `padding` 和 `truncation` 的作用与参数设置
  - 探讨中英文 Tokenization 的差异与常见问题
- `[案例3：预训练模型详解](src/basics/03_pretrained_models.py)`
  - 了解常见的预训练模型架构（如 BERT, GPT, T5, BART）及其特点
  - 学习如何根据任务需求选择合适的预训练模型
  - 掌握 `AutoModel` 系列类 (`AutoModel`, `AutoModelForSequenceClassification`, etc.) 的使用
- `[案例4：模型加载与保存](src/basics/04_model_loading_saving.py)`
  - 学习使用 `from_pretrained()` 方法从 Hugging Face Hub 或本地路径加载模型和 Tokenizer
  - 学习使用 `save_pretrained()` 方法将模型和 Tokenizer 保存到本地
  - 理解模型权重文件 (`pytorch_model.bin` 或 `tf_model.h5`) 和配置文件 (`config.json`)
- `[案例5：模型配置 Configuration](src/basics/05_configuration_objects.py)`
  - 了解 `AutoConfig` 和具体模型的 `Config` 对象（如 `BertConfig`）
  - 学习查看和修改模型配置参数（如隐藏层大小、注意力头数量）
  - 理解配置在模型初始化和行为中的作用

### 2. 模型应用 (src/models)
- `[案例1：文本分类](src/models/01_text_classification.py)`
  - 学习使用 `AutoModelForSequenceClassification` 进行文本分类任务（如情感分析、主题识别）
  - 掌握输入数据的准备和模型输出的解析
  - 实践微调预训练模型以适应特定分类任务
- `[案例2：命名实体识别 (NER)](src/models/02_named_entity_recognition.py)`
  - 学习使用 `AutoModelForTokenClassification` 进行 NER
  - 理解 Token 级别的分类任务
  - 掌握 NER 数据的处理和模型输出的对齐
- `[案例3：问答系统 (QA)](src/models/03_question_answering.py)`
  - 学习使用 `AutoModelForQuestionAnswering` 构建抽取式问答系统
  - 理解输入（问题和上下文）和输出（答案在上下文中的起止位置）
  - 实践处理长文本和无答案情况
- `[案例4：文本生成](src/models/04_text_generation.py)`
  - 学习使用 `AutoModelForCausalLM` (如 GPT-2) 和 `AutoModelForSeq2SeqLM` (如 T5, BART) 进行文本生成
  - 掌握 `generate()` 方法及其常用参数（如 `max_length`, `num_beams`, `do_sample`, `top_k`, `top_p`）
  - 实践不同的解码策略（Greedy Search, Beam Search, Sampling）
- `[案例5：机器翻译](src/models/05_machine_translation.py)`
  - 学习使用 `AutoModelForSeq2SeqLM` (如 MarianMT, T5) 进行机器翻译
  - 了解多语言模型和特定语言对模型的使用
  - 掌握翻译任务的数据准备和评估方法
- `[案例6：零样本/少样本分类](src/models/06_zero_shot_classification.py)`
  - 学习使用 `pipeline("zero-shot-classification")` 进行零样本分类
  - 理解其原理以及如何自定义候选标签
  - 探讨如何应用于自定义的文本分类任务，特别是在缺乏大量标注数据时

### 3. 数据处理 (src/datasets)
- `[案例1：Hugging Face Datasets 库入门](src/datasets/01_huggingface_datasets.py)`
  - 学习安装和使用 `datasets` 库
  - 掌握从 Hub 加载标准数据集 (`load_dataset`)
  - 学习数据集对象的常用操作（查看、选择、切分、迭代）
  - **本地数据集加载**：如何加载本地 CSV/JSON/TXT 文件
  - **数据集保存与导出**：如何将数据集保存为本地文件（如 `.arrow`、`.csv`）
  - **数据集探索与可视化**：使用 `features`、`info`、`head()`、`shuffle()`、`filter()` 等方法
  - **常见错误与调试**：加载数据集时的常见报错及解决方法
  - **与模型训练的衔接**：如何将 `datasets` 加载的数据直接用于模型训练
- `[案例2：数据预处理与 Tokenization](src/datasets/02_data_preprocessing.py)`
  - 学习使用 `map()` 方法对数据集进行批量 Tokenization 和预处理
  - 掌握如何将文本数据转换为模型可接受的输入格式 (`input_ids`, `attention_mask`, `token_type_ids`)
  - 理解 `batched=True` 的高效处理
- `[案例3：数据整理器 Data Collators](src/datasets/03_data_collators.py)`
  - 学习 `DataCollatorWithPadding` 的作用和使用，实现动态填充
  - 了解针对特定任务的数据整理器，如 `DataCollatorForLanguageModeling`
  - 学习如何在 `Trainer` 或 `DataLoader` 中使用 Data Collator
- `[案例4：自定义数据集](src/datasets/04_custom_datasets.py)`
  - 学习如何从本地文件（CSV, JSON, TXT）创建自定义数据集
  - 掌握使用 `Dataset.from_dict()` 或 `Dataset.from_generator()` 创建数据集
  - 实践将自定义数据集与 `Trainer` API 集成
- `[案例5：数据增强简介](src/datasets/05_data_augmentation.py)`
  - 了解文本数据增强的常见方法（如回译、同义词替换、随机插入/删除）
  - 探讨数据增强在提升模型泛化能力方面的作用
  - (可选) 实践使用第三方库进行简单的数据增强操作

### 4. 模型训练 (src/training)
- `[案例1：使用 Trainer API 进行微调](src/training/01_fine_tuning_with_trainer.py)`
  - 学习 Hugging Face `Trainer` API 的基本使用流程
  - 掌握如何定义模型、数据集、`TrainingArguments` 并开始训练
  - 理解训练循环的主要步骤和回调机制
- `[案例2：TrainingArguments 详解](src/training/02_training_arguments.py)`
  - 详细学习 `TrainingArguments` 中的常用参数及其作用
    - 输出目录、日志策略、评估策略、保存策略
    - 学习率、批大小、训练轮次、权重衰减
    - GPU 相关设置
- `[案例3：优化器与学习率调度器](src/training/03_optimizers_schedulers.py)`
  - 了解常用的优化器（如 AdamW）及其在 Transformers 中的应用
  - 学习不同类型的学习率调度器（如线性、余弦）及其配置
  - 理解它们在训练稳定性和模型性能方面的影响
- `[案例4：评估指标与模型评估](src/training/04_metrics_evaluation.py)`
  - 学习如何定义 `compute_metrics` 函数来计算评估指标 (如 accuracy, F1, precision, recall, perplexity)
  - 掌握使用 `evaluate` 库（前身为 `datasets.load_metric`）加载和使用标准评估指标
  - 理解如何在训练过程中和训练结束后评估模型性能
- `[案例5：自定义训练循环 (可选)](src/training/05_custom_training_loop.py)`
  - 了解何时以及为何需要自定义训练循环 (不使用 `Trainer`)
  - 学习使用 PyTorch/TensorFlow 从头开始编写训练和评估代码
  - 掌握手动处理梯度、优化器步骤、学习率调整等细节
- `[案例6：分布式训练简介](src/training/06_distributed_training.py)`
  - 了解分布式训练的基本概念（数据并行、模型并行）
  - 学习如何在 `Trainer` 中启用分布式训练（如 `torch.distributed.launch` 或 Accelerate）
  - 探讨分布式训练对加速大规模模型训练的重要性

### 5. 高级应用 (src/advanced)
- `[案例1：模型量化](src/advanced/01_model_quantization.py)`
  - 理解模型量化的基本原理（将 FP32 权重转换为 INT8）及其优势（减小模型体积、加速推理）
  - 学习动态量化和静态量化的概念与方法
  - 实践使用 PyTorch 或 ONNX Runtime 等工具对 Transformers 模型进行量化
- `[案例2：模型蒸馏](src/advanced/02_model_distillation.py)`
  - 理解知识蒸馏的原理：训练一个小（学生）模型来模仿一个大（教师）模型的行为
  - 学习常见的蒸馏方法和损失函数
  - 实践如何使用 Transformers 库或第三方工具实现模型蒸馏
- `[案例3：多任务学习](src/advanced/03_multi_task_learning.py)`
  - 理解多任务学习的概念：让一个模型同时学习和优化多个相关任务
  - 探讨不同多任务学习的架构和策略
  - (可选) 实践如何调整模型结构和训练过程以支持多任务
- `[案例4：跨语言迁移](src/advanced/04_cross_lingual_transfer.py)`
  - 了解跨语言模型（如 XLM-R）的原理和应用
  - 学习如何利用在高资源语言上预训练的模型来提升低资源语言任务的性能 (零样本或少样本跨语言迁移)
  - 实践微调跨语言模型于特定下游任务
- `[案例5：模型部署与推理优化](src/advanced/05_model_deployment.py)`
  - 了解将 Transformers 模型部署到生产环境的常见方案（如 ONNX, TorchScript, Hugging Face Inference Endpoints, BentoML, Triton Inference Server）
  - 学习推理优化技术（如使用 ONNX Runtime, TensorRT, JIT compilation, 混合精度推理）
  - 探讨如何平衡推理速度、模型大小和准确性
- `[案例6：使用 Accelerate 进行高效训练](src/advanced/06_accelerate_integration.py)`
  - 学习 Hugging Face `Accelerate` 库，简化 PyTorch 训练脚本在不同硬件（单 GPU、多 GPU、TPU）上的移植
  - 掌握 `Accelerator` 对象的核心用法，自动处理设备分配和分布式训练设置
  - 实践将现有的 PyTorch 训练脚本用 `Accelerate` 进行重构

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
