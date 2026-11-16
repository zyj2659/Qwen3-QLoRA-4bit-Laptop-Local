import os

# ============================================================
# 0. 环境配置
# 必须在导入 torch 和 unsloth 之前设置
# ============================================================

os.environ["HF_HOME"] = r"D:\ComVision\code\qw_sailinna"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["UNSLOTH_LLAMA_CPP_PATH"] = (
    r"D:\ComVision\env\the_sai\llama_cpp"
)

# ============================================================
# 1. 导入依赖
# ============================================================

import unsloth
import torch

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


# ============================================================
# 2. 全局参数
# ============================================================

# 修改为你自己的 JSONL 文件路径
DATA_PATH = (
r"data/sample.jsonl"
)

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

MAX_SEQ_LENGTH = 1024

# 第一次运行保持 False
# 确认检查全部通过后改成 True
DO_TRAIN = True

# 输出目录
LORA_OUTPUT = "selena_lora"
GGUF_OUTPUT = "selena_gguf"


# ============================================================
# 3. 加载模型
# ============================================================

print("\n========== 1. 加载模型 ==========")

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"找不到数据文件，请检查 DATA_PATH：\n{DATA_PATH}"
    )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("模型加载成功！")


# ============================================================
# 4. 修改 Chat Template
# 去掉没有实际思考内容时自动添加的空 Thinking 标签
# ============================================================

print("\n========== 2. 修改 Chat Template ==========")

old_rule = (
    "{%- if loop.last or (not loop.last and reasoning_content) %}"
)

new_rule = (
    "{%- if reasoning_content %}"
)

if not isinstance(tokenizer.chat_template, str):
    raise RuntimeError(
        "当前 tokenizer.chat_template 不是字符串，请先检查模板结构。"
    )

count = tokenizer.chat_template.count(old_rule)

if count != 1:
    raise RuntimeError(
        f"模板规则匹配数量异常：{count}，停止修改。"
    )

tokenizer.chat_template = tokenizer.chat_template.replace(
    old_rule,
    new_rule,
)

print("Chat Template 修改成功！")


# ============================================================
# 5. 配置 LoRA
# ============================================================

print("\n========== 3. 配置 LoRA ==========")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=3407,
)

print("LoRA 配置完成！")


# ============================================================
# 6. 读取 JSONL 数据
# ============================================================

print("\n========== 4. 加载数据 ==========")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

# 只保留真正用于训练的 messages
dataset = dataset.select_columns(["messages"])

print("数据条数：", len(dataset))
print("数据字段：", dataset.column_names)



# ============================================================
# 7. 检查 Chat Template 输出
# ============================================================

print("\n========== 5. 检查对话格式 ==========")

messages = dataset[0]["messages"]

test_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print(test_text)

if "<think>" in test_text or "</think>" in test_text:
    raise RuntimeError(
        "Chat Template 修改失败，仍然存在 Thinking 标签。"
    )

print("\n对话格式检查通过！")


# ============================================================
# 8. 检查所有数据的 Token 长度
# ============================================================

print("\n========== 6. 检查 Token 长度 ==========")

token_lengths = []

for i, example in enumerate(dataset):

    token_ids = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )

    token_lengths.append(len(token_ids))


print("样本数量：", len(token_lengths))
print("最短 Token 数：", min(token_lengths))

print(
    "平均 Token 数：",
    sum(token_lengths) / len(token_lengths)
)

print("最长 Token 数：", max(token_lengths))

over_limit = [
    (i, length)
    for i, length in enumerate(token_lengths)
    if length > MAX_SEQ_LENGTH
]

print("超过1024 Token的样本数：", len(over_limit))

if over_limit:

    print("超长样本：", over_limit)

    raise RuntimeError(
        "存在超过最大序列长度的样本，请先处理。"
    )

print("Token 长度检查通过！")


# ============================================================
# 9. 构建 SFTTrainer
# ============================================================

print("\n========== 7. 构建 Trainer ==========")

trainer = SFTTrainer(

    model=model,

    processing_class=tokenizer,

    train_dataset=dataset,

    args=SFTConfig(

        output_dir="赛琳娜",

        # 最大序列长度
        max_length=MAX_SEQ_LENGTH,

        # Batch 设置
        per_device_train_batch_size=2,

        gradient_accumulation_steps=8,

        # 训练轮数
        num_train_epochs=10,

        warmup_steps=5,

        # 学习率
        learning_rate=1e-5,

        # 精度
        fp16=not torch.cuda.is_bf16_supported(),

        bf16=torch.cuda.is_bf16_supported(),

        # 日志
        logging_steps=1,

        # 优化器
        optim="adamw_8bit",

        weight_decay=0.01,

        lr_scheduler_type="linear",

        seed=3407,

        # 核心配置：只计算 assistant 的 Loss
        assistant_only_loss=True,

        # 不打包多个样本
        packing=False,

    ),
)

print("Trainer 构建完成！")


# ============================================================
# 10. 检查 Loss Mask
# ============================================================

print("\n========== 8. 检查 Loss Mask ==========")

print(
    "Trainer 数据字段：",
    trainer.train_dataset.column_names
)

# 获取一个真实训练 batch
batch = next(iter(trainer.get_train_dataloader()))

if "labels" not in batch:
    raise RuntimeError(
        "没有找到 labels，assistant-only Loss 可能没有生效。"
    )

input_ids = batch["input_ids"][0].tolist()

labels = batch["labels"][0].tolist()


# 统计 Token
total_tokens = len(labels)

train_tokens = sum(
    label != -100 for label in labels
)

masked_tokens = sum(
    label == -100 for label in labels
)


print("总 Token 数：", total_tokens)

print("参与 Loss 的 Token 数：", train_tokens)

print("不参与 Loss 的 Token 数：", masked_tokens)


# ============================================================
# 11. 解码实际参与 Loss 的文本
# ============================================================

print("\n========== 9. 检查实际训练目标 ==========")

supervised_ids = [

    token_id

    for token_id, label in zip(input_ids, labels)

    if label != -100

]

supervised_text = tokenizer.decode(
    supervised_ids,
    skip_special_tokens=False,
)

print(supervised_text)


# 检查是否还存在 Thinking 标签
if "<think>" in supervised_text or "</think>" in supervised_text:

    raise RuntimeError(
        "实际参与 Loss 的文本中仍然存在 Thinking 标签！"
    )


# 检查是否有参与训练的 Token
if train_tokens == 0:

    raise RuntimeError(
        "参与 Loss 的 Token 数为0，不能开始训练。"
    )


print("\nLoss Mask 检查通过！")


# ============================================================
# 12. 正式训练
# ============================================================

if not DO_TRAIN:

    print("\n======================================")
    print("所有检查完成，目前没有开始训练。")
    print("确认结果无误后，将 DO_TRAIN 改为 True。")
    print("======================================")

else:

    print("\n========== 10. 正式开始训练 ==========")

    trainer_stats = trainer.train()

    print("\n训练完成！")


    # ========================================================
    # 13. 保存 LoRA
    # ========================================================

    print("\n========== 11. 保存 LoRA ==========")

    model.save_pretrained(LORA_OUTPUT)

    tokenizer.save_pretrained(LORA_OUTPUT)

    print("LoRA 保存完成！")


    # ========================================================
    # 14. 导出 GGUF
    # ========================================================

    print("\n========== 12. 导出 GGUF ==========")

    model.save_pretrained_gguf(
        GGUF_OUTPUT,
        tokenizer,
        quantization_method="q4_k_m",
    )

    print("\n======================================")
    print("赛琳娜模型训练与导出全部完成！")
    print("======================================")