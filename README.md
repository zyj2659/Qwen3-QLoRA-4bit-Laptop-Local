# 二次元人物对话模型 LoRA 微调

基于 [Unsloth](https://github.com/unslothai/unsloth) + LoRA，对 **Qwen3-4B-Instruct-2507** 做二次元人物对话风格的微调，并在**消费级显卡（家用/游戏本）**上完成训练与 GGUF 导出。

训练数据自备：仓库只提供训练代码和一个对话格式样例，你可以用任意原创角色或已获授权的人设，借助大模型批量扩充数据。

## 特性

- 单卡即可训练：实测 **RTX 3060 Laptop 6GB / Windows 11** 可完成 4B 模型 4-bit QLoRA 训练
- 使用 Unsloth 加速，显存占用低，训练速度快
- 训练前内置 6 项自动检查（数据、模板、token 长度、loss mask 等），避免白跑
- 支持多轮对话，只对 assistant 回复计算 loss
- 训练完成后自动保存 LoRA，并可导出 `q4_k_m` 量化 GGUF

## 仓库内容说明

本仓库**只包含训练代码**。

| 内容 | 说明 |
|---|---|
| 训练数据 | 由使用者自备，仓库仅提供格式样例 |
| LoRA、GGUF、checkpoint | 训练产物与模型权重 | 可通过镜像链接下载
| `*.gguf`、`*.safetensors`、`*.pt` 等 | 权重文件 | 最后导出的文件，可以导入到ollama

## 目录结构

```
.
├── train.py             # 训练 + 预检查 + 导出脚本
├── requirements.txt     # 依赖（与实测 conda 环境一致）
├── data/
│   └── sample.jsonl     # 一段对话样例（数据格式参考）
└── README.md
```

## 环境要求

- Windows 11 / Linux
- Python **3.12**
- CUDA **12.1**（PyTorch `2.5.1+cu121`）
- NVIDIA **消费级显卡**（家用/游戏本），显存 **≥ 6 GB**
  - 实测：RTX 3060 Laptop 6GB，Windows 11，可完成训练
- conda 环境名：`the_sai`
- 磁盘：模型缓存约 3 GB，GGUF 导出额外需要约 8 GB

`requirements.txt` 中的版本取自实际训练环境：

| 组件 | 版本 |
|---|---|
| torch | 2.5.1+cu121 |
| transformers | 5.17.0 |
| trl | 1.15.0 |
| datasets | 4.8.5 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| unsloth / unsloth_zoo | 2026.11.2 |
| triton-windows | 3.9.0.post30 |

## 安装

```bash
git clone <your-repo-url>
cd <repo-name>

conda create -n My_LLM python=3.12 -y
conda activate My_LLM

pip install -r requirements.txt
```

> `torch==2.5.1+cu121` 需要 PyTorch 官方 CUDA 12.1 索引，`requirements.txt` 首行已通过
> `--extra-index-url` 配置好。

## 数据格式

`train.py` 读取 JSONL，每行一个样本，只使用 `messages` 字段，交给 tokenizer 的 chat template 渲染。支持单轮和多轮：

```json
{"messages": [{"role": "user", "content": "今天也辛苦了，早点休息吧。"}, {"role": "assistant", "content": "（轻轻摇头）还不行呢……今天的曲子还差最后一段。不过，谢谢你来陪我。"}]}
```

上面这条样例也放在 `examples/sample.jsonl` 中，可以直接复制成自己的数据文件，或作为扩充的种子。把数据文件路径填到 `train.py` 顶部的 `DATA_PATH`，格式保持 `messages` 数组即可。角色名、语气、世界观都由你的数据决定。

## 用大模型扩充数据

只有一条样例时，可以用下面这段提示词连同样例一起交给大模型，批量生成同风格的对话，再人工抽检。
或者说galgame或者小说文本等，可以喂给大模型，让他帮忙生成扩充对话数据，下面给出样例

```text
你是一个对话数据生成器。请根据下面的角色设定和示例，生成 {N} 条符合该角色风格的对话。

【角色设定】
- 名称：
- 身份：
- 性格：
- 说话习惯（语气、口头禅、句长、是否使用括号动作描写）：

【示例对话】
{"messages": [{"role": "user", "content": "今天也辛苦了，早点休息吧。"}, {"role": "assistant", "content": "（轻轻摇头）还不行呢……今天的曲子还差最后一段。不过，谢谢你来陪我。"}]}

【生成要求】
1. 输出 JSONL：每行一个 JSON 对象，字段只有 messages。
2. messages 为 user/assistant 交替，1～3 轮对话。
3. 严格保持第一人称和角色语气，不要输出设定说明、旁白解释或 meta 信息。
4. 每条 assistant 回复控制在 100 字以内，口语化，允许括号动作描写。
5. 场景多样化：日常问候、闲聊、被关心、一起做事、轻微情绪波动等，不要重复同一句式。
6. 不要复述或改写角色设定本身，不要出现真实人物、作品名或现实品牌。
7. 只输出 JSONL 文本，不要代码块以外的解释。

现在开始生成：
```

建议：
- 先一次生成 20～50 条，人工检查风格是否稳定，再扩大规模；
- 风格微调通常 200～1000 条高质量对话即可见效，质量比数量重要；
- 生成后逐行校验 JSON，避免大模型输出多余文字导致加载失败。

## 使用

训练前修改 `train.py` 顶部与运行环境相关的路径：

```python
os.environ["HF_HOME"] = r"<你的 HuggingFace 缓存目录>"
os.environ["UNSLOTH_LLAMA_CPP_PATH"] = r"<你的 llama.cpp 目录>"
DATA_PATH = r"<你的 JSONL 数据文件>"
```

然后运行：

```bash
python train.py
```

建议第一次把 `DO_TRAIN` 设为 `False`，脚本会依次执行以下检查，全部通过后再改为 `True` 正式训练：

1. 数据文件是否存在
2. 修改 Qwen3 chat template，去掉无 reasoning 内容时自动插入的空 `<think>` 标签，并校验替换次数
3. 打印首条样本的 chat template 渲染结果
4. 逐条统计 token 长度，任一超过 `MAX_SEQ_LENGTH=1024` 直接报错
5. 构建 `SFTTrainer`，检查 `labels` 中 assistant-only loss 的 mask 是否正确
6. 解码实际参与 loss 的文本，确认其中没有 `<think>` 标签

## 训练配置

| 参数 | 值 |
|---|---|
| 基座模型 | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| 量化 | 4-bit（QLoRA） |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0.05 |
| target modules | q、k、v、o、gate、up、down proj |
| 最大序列长度 | 1024 |
| batch size × 梯度累积 | 2 × 8 |
| epochs | 10 |
| learning rate | 1e-5（linear scheduler，warmup 5） |
| 优化器 | adamw_8bit |
| 精度 | bf16（不支持时 fp16） |
| loss | `assistant_only_loss=True` |
| packing | `False` |
| seed | 3407 |

> 训练脚本设置了 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像）。如在其他网络环境使用，请删除或修改该行。

## 输出文件

脚本会把三类产物写到三个目录，路径由 `train.py` 中的 `output_dir`、`LORA_OUTPUT`、`GGUF_OUTPUT` 控制：

| 变量 | 内容 |
|---|---|
| `output_dir` | `SFTTrainer` 的 checkpoint 与日志 |
| `LORA_OUTPUT` | 最终 LoRA adapter 与 tokenizer |
| `GGUF_OUTPUT` | 导出的 GGUF（`q4_k_m`） |


### GGUF 导出

`train.py` 末尾调用 `model.save_pretrained_gguf(..., quantization_method="q4_k_m")`，依赖 llama.cpp：

- 将 `UNSLOTH_LLAMA_CPP_PATH` 指向本机 llama.cpp 目录（需包含 `convert_hf_to_gguf.py` 与对应二进制）；
- 不需要 GGUF 时，注释掉脚本最后的导出段即可，LoRA 权重仍会正常保存。

## 说明

- 超参面向 6 GB 显存的消费级显卡，数据量较大时可适当提高 batch 或降低 epochs。
- 脚本中的路径是作者本机路径，迁移到其他机器时务必修改。
- 若数据中存在超过 1024 token 的样本，脚本会直接报错，需要先截断或改写。

## 数据与版权

- 本仓库不附带任何训练语料与模型权重，使用者需自备数据。
- 角色形象、作品名称及相关素材的版权归各自权利人所有；请确保你使用的数据来源合法、已获授权，或仅用于个人学习研究。
- 生成与发布模型权重前，请自行确认是否符合原权利人的使用条款与所在地区的法律法规。
