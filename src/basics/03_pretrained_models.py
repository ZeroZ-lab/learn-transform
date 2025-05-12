"""
案例3：预训练模型详解

本案例介绍主流的预训练 Transformer 模型架构，包括 BERT、GPT、T5、BART 等。

学习目标：
- 了解常见的预训练模型架构及其特点
- 学习如何根据任务需求选择合适的预训练模型
- 掌握 AutoModel 系列类的使用方法
- 对比不同模型的输入输出格式和适用场景

【典型应用场景】
1. BERT：文本分类、序列标注、问答、特征提取
2. GPT：文本生成、对话系统、补全任务
3. T5/BART：文本生成、摘要、翻译、问答

企业级最佳实践：
- 优先使用 Hugging Face Hub 上的主流预训练模型
- 通过 AutoModel/AutoTokenizer 实现模型与任务解耦
- 注意模型输入输出格式的差异，合理选择模型

"""

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline # 引入 pipeline 便于快速推理
)
import torch # 引入 torch 处理模型输出

def show_model_info(model_name, task_class):
    print(f"\n加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = task_class.from_pretrained(model_name)
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Model type:", model.config.model_type)
    # 注意：对于大型模型，计算参数数量可能会比较慢或消耗大量内存
    # print("Model parameters:", model.num_parameters())
    return tokenizer, model

def main():
    print("--- 丰富案例：加载与推理示例 ---")

    # 1. BERT: 适合分类、序列标注、问答
    # 1.1 文本分类 (Sequence Classification)
    print("\n--- 1.1 文本分类 (BERT) ---")
    model_name_cls = "bert-base-uncased"
    tokenizer_cls, model_cls = show_model_info(model_name_cls, AutoModelForSequenceClassification)
    
    # 推理示例
    text_cls = "This movie is absolutely fantastic and I loved it!"
    print(f"输入文本: '{text_cls}'")
    inputs_cls = tokenizer_cls(text_cls, return_tensors="pt") # 返回 PyTorch 张量
    
    with torch.no_grad(): # 推理阶段不需要计算梯度
        outputs_cls = model_cls(**inputs_cls)
    
    # 模型输出通常是 logits，需要通过 softmax 转换为概率，并找到概率最大的类别
    predictions_cls = torch.softmax(outputs_cls.logits, dim=-1)
    # 注意：bert-base-uncased + AutoModelForSequenceClassification 默认只有两个类别 (0, 1)
    # 它们可能代表积极/消极等，具体含义取决于模型的微调方式。
    # 这里的输出只是原始模型的输出，没有经过特定任务的微调，所以类别标签无实际意义。
    # 实际应用中，会加载在特定数据集上微调过的模型。
    predicted_class_id_cls = predictions_cls.argmax().item()
    print(f"模型输出 (logits): {outputs_cls.logits}")
    print(f"预测类别 ID: {predicted_class_id_cls}") # 实际使用时需要映射到具体的类别标签

    # 1.2 序列标注 (Token Classification - 例如 NER)
    print("\n--- 1.2 序列标注 (BERT - NER) ---")
    model_name_ner = "bert-base-cased" # 通常 NER 使用 cased 模型
    tokenizer_ner, model_ner = show_model_info(model_name_ner, AutoModelForTokenClassification)

    # 推理示例
    text_ner = "Paris is the capital of France."
    print(f"输入文本: '{text_ner}'")
    inputs_ner = tokenizer_ner(text_ner, return_tensors="pt")

    with torch.no_grad():
        outputs_ner = model_ner(**inputs_ner)

    predictions_ner = torch.argmax(outputs_ner.logits, dim=-1)

    # 将预测的类别 ID 映射回标签
    # 注意：这里加载的是基础模型，其 config.id2label 可能不存在或不包含实际的 NER 标签
    # 实际应用中，会加载在 NER 数据集上微调过的模型，其 config 会有 id2label 映射
    # 我们先尝试获取标签，如果不存在则只打印 ID
    id2label = model_ner.config.id2label if hasattr(model_ner.config, 'id2label') else None

    print("输入 Tokens 和预测 ID:")
    tokens = tokenizer_ner.convert_ids_to_tokens(inputs_ner["input_ids"][0])
    predicted_ids = predictions_ner[0].tolist()

    for token, pred_id in zip(tokens, predicted_ids):
        label = id2label[pred_id] if id2label else f"ID:{pred_id}"
        print(f"{token}: {label}")

    # 1.3 问答 (Question Answering)
    print("\n--- 1.3 问答 (BERT - SQuAD) ---")
    model_name_qa = "bert-large-uncased-whole-word-masking-finetuned-squad" # 微调过的问答模型
    tokenizer_qa, model_qa = show_model_info(model_name_qa, AutoModelForQuestionAnswering)

    # 推理示例
    question_qa = "What is the capital of France?"
    context_qa = "Paris is the capital of France. It is a major European city."
    print(f"问题: '{question_qa}'")
    print(f"上下文: '{context_qa}'")

    inputs_qa = tokenizer_qa(question_qa, context_qa, return_tensors="pt")

    with torch.no_grad():
        outputs_qa = model_qa(**inputs_qa)

    # 问答模型输出的是答案开始和结束位置的 logits
    answer_start_index = outputs_qa.start_logits.argmax()
    answer_end_index = outputs_qa.end_logits.argmax()

    # 从上下文对应的 token 中提取答案
    # 需要注意特殊 token ([CLS], [SEP]) 的位置
    input_ids_qa = inputs_qa["input_ids"][0].tolist()
    tokens_qa = tokenizer_qa.convert_ids_to_tokens(input_ids_qa)

    # 找到答案 token 跨度
    answer_tokens = tokens_qa[answer_start_index:answer_end_index+1]
    # 将 token 合并回原始文本 (处理 ## 等子词标记)
    answer = tokenizer_qa.decode(tokenizer_qa.convert_tokens_to_ids(answer_tokens))

    print(f"预测答案: '{answer}'")


    # 2. GPT-2: 适合文本生成 (Causal Language Modeling)
    print("\n--- 2. 文本生成 (GPT-2) ---")
    model_name_gen = "gpt2"
    tokenizer_gen, model_gen = show_model_info(model_name_gen, AutoModelForCausalLM)

    # 推理示例
    prompt_text = "Once upon a time,"
    print(f"输入提示文本: '{prompt_text}'")

    inputs_gen = tokenizer_gen(prompt_text, return_tensors="pt")

    # 使用 generate 方法进行文本生成
    # max_length 控制生成的最大 token 数
    # num_return_sequences 控制生成多少个不同的序列
    # no_repeat_ngram_size 避免重复的 n-gram
    # early_stopping 在达到 EOS token 时停止生成
    print("正在生成文本...")
    generated_outputs = model_gen.generate(
        inputs_gen["input_ids"],
        max_length=50, # 生成到总共 50 个 token (包括 prompt)
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer_gen.eos_token_id # 对于 GPT-2，pad_token_id 通常设置为 eos_token_id
    )

    # 解码生成的 token ID 为文本
    generated_text = tokenizer_gen.decode(generated_outputs[0], skip_special_tokens=True)
    print(f"生成的文本: '{generated_text}'")


    # 3. T5: 适合生成、翻译、摘要 (Seq2Seq Language Modeling)
    print("\n--- 3. 序列到序列 (T5 - 翻译示例) ---")
    model_name_t5 = "t5-small" # small 版本模型较小，方便演示
    tokenizer_t5, model_t5 = show_model_info(model_name_t5, AutoModelForSeq2SeqLM)

    # 推理示例 (翻译)
    # T5 模型通常需要在输入前添加任务前缀，例如 "translate English to German: "
    text_t5 = "translate English to German: The house is wonderful."
    print(f"输入文本 (带任务前缀): '{text_t5}'")

    inputs_t5 = tokenizer_t5(text_t5, return_tensors="pt")

    # 使用 generate 方法进行生成
    print("正在进行翻译...")
    generated_outputs_t5 = model_t5.generate(
        inputs_t5["input_ids"],
        max_length=50,
        num_return_sequences=1,
        early_stopping=True
    )

    # 解码生成的 token ID
    translated_text = tokenizer_t5.decode(generated_outputs_t5[0], skip_special_tokens=True)
    print(f"翻译结果: '{translated_text}'")


    # 4. BART: 适合生成、摘要、翻译 (Seq2Seq Language Modeling)
    print("\n--- 4. 序列到序列 (BART - 摘要示例) ---")
    model_name_bart = "facebook/bart-base"
    tokenizer_bart, model_bart = show_model_info(model_name_bart, AutoModelForSeq2SeqLM)

    # 推理示例 (摘要)
    # BART 基础模型通常没有特定的任务前缀，直接输入原文即可
    text_bart = """
    New York (CNN)When asked about their biggest fears regarding artificial intelligence, most people express concerns about superintelligent robots taking over the world.
    But experts say the more immediate risks are subtle, mundane and already here. AI is increasingly being used in sensitive applications, such as screening job applicants, determining loan eligibility and even predicting crime.
    These systems can perpetuate and even amplify existing societal biases if they are not carefully designed and monitored. For example, if an AI system is trained on historical data where a particular demographic group was unfairly denied loans, the system might learn to deny loans to members of that group, even if it was not explicitly programmed to do so.
    """
    print(f"输入长文本 (用于摘要): '{text_bart[:100]}...'") # 只打印前100个字符作为示例

    inputs_bart = tokenizer_bart(text_bart, return_tensors="pt", max_length=1024, truncation=True) # 摘要通常输入较长

    # 使用 generate 方法进行生成
    print("正在生成摘要...")
    generated_outputs_bart = model_bart.generate(
        inputs_bart["input_ids"],
        max_length=150, # 摘要的期望长度
        min_length=40,
        num_beams=4, # 使用 Beam Search 提高摘要质量
        early_stopping=True
    )

    # 解码生成的 token ID
    summary_text = tokenizer_bart.decode(generated_outputs_bart[0], skip_special_tokens=True)
    print(f"生成的摘要: '{summary_text}'")

    print("\n--- 丰富案例：加载与推理示例 结束 ---")


if __name__ == "__main__":
    main() 