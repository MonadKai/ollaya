# Elephant-1 模型契约与参考运行时核查

核查日期：2026-11-06。对象固定为
[`baselabs/elephant-1@b3f8a93462fb1051a9de031e94a7ff380e254072`](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072)
及作者 [`setup.sh`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/setup.sh)
指定的 OneJev `81ce62f1597c91e46767d4d02d6ac2e18534fe94` +
[`serve.patch`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)。
以下是移植 Ollaya 时必须保持的模型行为，而不是通用 Qwen3.5 聊天推理约定。
本次没有运行完整模型；本机 18.82 GB 权重仍在下载，权重头部观察须待完整 SHA-256 回执后才成为可用资产的证据。

## 固定检查点的事实

| 项 | 证据和含义 |
| --- | --- |
| 架构 | [`config.json`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/config.json) 为 `Qwen3_5ForConditionalGeneration`，`qwen3_5`，32 个文本层，每四层是三层 linear attention 加一层 full attention；视觉塔为 27 层。模型卡称从 Qwen3.5-9B 全量微调语言模型、视觉编码器冻结，输入可含文本和图像。[模型卡](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/README.md) |
| 注意力 | `modeling_overrides.attention` 指定 full attention 为 `block_bidirectional`，linear attention 为 `causal`。不能把模型当成全因果 Qwen3.5，也不能让 Gated DeltaNet 层双向化。[配置](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/config.json)、[作者技术报告 §2.3](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) |
| 读出 | [`readout.json`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/readout.json) 是 `slot_lm`，`n_slots=256`，`head_dim=256`。这里 `head_dim` 是兼容读出配置字段；本检查点没有 `readout_head.pt`，不使用训练出的 pointer/head 读出。答案来自 LM head 的字母槽位 logits，而非生成文本或额外分类头。[补丁 readout.py](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch) |
| 权重 | 固定仓库 15 文件，主权重 `model.safetensors` 18,819,722,392 字节，HF LFS SHA-256 `682f02348dfe4617dd21bba2915c12e805de8cf16327a61e895a831ea681d0ba`。[仓库清单](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072) 本机未完成权重的 safetensors 头部可解析出 760 个 BF16 tensor（语言 426、视觉 333、LM head 1），但尚未核完整文件哈希。 |
| 校准 | 仓库没有 `calibration.json`；补丁只在本地文件存在时加载温度表，缺失时 OneJev 的 `Calibration.temperature` 回退到 1.0。这里的“校准概率”主要来自训练目标，不能自行补一个后处理温度。[补丁 engine.py](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)、[OneJev 校准实现](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/calibrate.py#L12-L40)、[作者技术报告 §2.6](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models) |

本机用仓库固定的 `tokenizer.json` 和 OneJev
[`SLOT_LABELS`](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/prompt.py#L17-L25)
实测：256 个 A–Z/双字母标签全部是单 token，`decode(encode(label)) == label`。
这是此 tokenizer 的观察值；移植时仍应作为加载期断言。

## 计算与输入输出契约

一次请求给出共享 state 和若干问题。每个问题是 `noul`（是/否）、`choice`
（1–255 个命名选项）或 `score`（有序等级）；state 可以是字符串或 JSON 值。
OneJev 的 [`schema.py`](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/schema.py#L12-L78)
与 [`prompt.py`](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/prompt.py#L57-L166)
固定系统指令、`<state>` 区块、问题/选项顺序与答案位置；tokenizer 的
[`chat_template.jinja`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/chat_template.jinja)
在 `enable_thinking=False` 下将它们转成最终 token 序列。模型看到的是这些精确 token，
不是普通聊天消息中的问题描述。选项 id 不进入提示文本；选项名称、描述、顺序会进入。

设共享前缀为系统提示加 state，后缀为某个问题、选项和答案位置。
full attention 层的可见规则是 `block(key) <= block(query)`：前缀内部双向、后缀内部双向，
后缀可见前缀，前缀不可见后缀；linear attention 层始终因果。
因此前缀计算结果可在所有问题间复用，后缀必须包含该问题**全部**选项，
读答案位置的最终 LM head logits，再只取对应槽位、按选项集合 softmax。
这决定了缓存切点、mask、position ids、option 排列和多问题分支都是数值语义的一部分。
[作者技术报告 §2.2–2.3](https://labs.baseten.co/articles/mammoth-1-and-elephant-1-frontier-decision-models)、
[补丁 readout.py / engine.py](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)

OneJev 先对每一分支完整渲染并编码，寻找不越过 state 边界的共同 token 前缀，
并校验非因果 block 边界和实际缓存切点一致；顺序模式复制前缀缓存，批量模式在 batch 维复制
full-attention KV 与 linear-attention convolution/recurrent states。
默认 `fork_mode=auto` 会用一个合成请求比较顺序/批量概率，超过 `0.05` 则退回顺序；
这只是结构性检查，不能作为 Ollaya 的精度验收阈值。
[OneJev 前缀及 fork 实现](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py#L176-L265)、
[作者补丁](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)

输出不采样 token。`noul` 返回 P(yes)；`choice` 返回每个原选项概率、最大概率选项及置信度；
`score` 返回等级概率、概率加权的期望等级、legend 及置信度。OneJev 组装时归一化并把数值
舍入到六位小数；它可做选项循环移位 debias，但默认 `debias=1`。
[OneJev 答案组装](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/answers.py#L12-L89)、
[OneJev decide](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py#L326-L427)

模型卡写明训练序列上限 16,384 token、超长请求拒绝；基础模型配置却有
`max_position_embeddings=262144`，OneJev `DecisionEngine` 默认分支/请求限额为
32,768/65,536。这三者不等价。建立参考结果时应保守限制到 16,384，
再向作者或运行结果核实精确的多问题总量限制，不能把底模位置上限当成模型训练上限。
[模型卡](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/README.md)、
[配置](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/config.json)、
[OneJev 限额](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py#L63-L92)

## 图像：检查点能力与公开自托管实现的差距

模型卡宣称文本和图像输入，仓库的视觉塔 tensor、`vision_config`、
[`processor_config.json`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/processor_config.json)
也支持这一点。OneJev 原有
[`MMDecisionEngine`](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/mm_engine.py#L315-L435)
可解析 state 中按序引用的 `<image:N>`/`<video:N>`，由 processor 展开视觉 token，
计算多模态 RoPE 位置，并把视觉编码放入共享前缀。
[媒体输入约定](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/media.py#L1-L96)、
[多模态前缀实现](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/mm_engine.py#L501-L559)

但固定版模型卡的自托管示例调用的是**文本** `qev.engine.DecisionEngine`；
`engine.py` 包装器也调用它。
[`serve.patch`](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)
修改 `qev/engine.py`、`qev/prompt.py`、新增 `qev/readout.py` 和训练/评估文件，
**没有修改 `qev/mm_engine.py`**。固定版 `MMDecisionEngine` 仍走自己的
[`_run_single`/`_run_sequential`/`_run_batched`](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/mm_engine.py#L561-L679)
且其 `__init__` 没有调用文本父类初始化，故作者补丁新增的 `Readout.attach`
不会自动安装到它；公开代码未见 Elephant-1 的 block mask 接入，多模态执行从源码看仍走默认因果路径。
因此“权重含视觉塔、作者宣称图像能力”有证据，而“公开自托管命令能给出正确的图像概率”
**尚无可复现证据**；不能以通用 `MMDecisionEngine` 输出作为图像 parity 真值。
若要支持图像，先从作者训练评估路径（补丁中的 `train/common.py::batch_scores`
和 `train/eval_mm.py`）或作者确认的服务获取固定输入的参考结果，
再验证视觉预处理、M-RoPE、block mask 和槽位概率。

另一个可复现性陷阱：补丁中 `DecisionEngine` 用
`local = Path(model_path).exists()` 判定来源，并仅在 `local` 时调用
`ReadoutConfig.load(model_path)` 与读取 `calibration.json`。
模型卡示例传入仓库名 `baselabs/elephant-1` 时，若本地没有同名目录，
会使用 `ReadoutConfig()` 的默认**因果**注意力，和检查点 `modeling_overrides` 不一致。
这由固定版[原始 OneJev 初始化](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py#L82-L111)
与[作者补丁初始化](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)直接推出，
需要运行时检查，不应把示例命令未经检查地当作基准。
本机 Ollaya 转换环境中的 Transformers 5.17 已有 `qwen3_5_text` 的
`language_model` 前缀转换映射，因此文本类从多模态权重读取**有源码依据**；
作者约束的是 Transformers 5.19，仍须在该版本实际检查 missing/unexpected keys，
不能只根据类名判断加载成功。

## 建立文本参考结果的安全入口

待本地固定快照的 15 文件与权重 SHA-256 全部校验通过后，在作者指定的
PyTorch/CUDA 镜像与 `setup.sh` 固定依赖环境中，**传绝对本地路径**并检查加载信息：

```python
from qev.engine import DecisionEngine

model_dir = "/opt/huggingface/hub/models--baselabs--elephant-1/snapshots/b3f8a93462fb1051a9de031e94a7ff380e254072"
engine = DecisionEngine(model_dir, device="cuda:0", fork_mode="sequential",
                        max_branch_tokens=16384, max_request_tokens=16384)
assert engine.info.extra["attention"] == "noncausal"
assert engine.info.extra["readout"] == "slot_lm"
assert engine.info.extra["calibration"] == "none"
```

这个入口是从公开代码推导的**待实测**参考方案；若完整多模态检查点无法被
`Qwen3_5ForCausalLM` 正确装入，或出现未加载权重警告，应停止生成 goldens，
先修复/确认作者文本加载路径。不要通过忽略权重缺失、切换成因果注意力或直接用
`transformers.pipeline("text-classification")` 绕过。`fork_mode="sequential"`
先隔离分支复制的数值差异，随后再单独验收批量分支。
[作者自托管依赖](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/setup.sh)、
[作者引擎入口](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/engine.py)

## 必须补齐的证据

1. 完整权重 SHA-256、15 文件清单及作者固定环境的真实加载结果；确认无缺失/意外权重键。
2. 作者顺序模式的文本 goldens：`noul`、`choice`、`score`、1/26/27/255 选项、JSON state、长 state、拒绝超限、多问题共享前缀；记录渲染 token、槽位 logits、softmax 概率及答案。
3. 在同一输入上比较作者单次完整 forward、前缀加顺序后缀和批量 fork；分别记录 BF16 舍入误差，不能把作者加载期的 0.05 容差直接作为移植验收值。
4. 图像单独建立可复现的作者真值，至少覆盖一张和多张图、图文顺序、视觉 token 展开与 M-RoPE；在公开自托管路径证实正确前，图像应保持未验证状态。

以上检查项由[作者补丁](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)、
[OneJev 文本引擎](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/engine.py)、
[OneJev 多模态引擎](https://github.com/OmniJev/OneJev/blob/81ce62f1597c91e46767d4d02d6ac2e18534fe94/qev/mm_engine.py)
的计算路径推导，不是本轮已通过的模型验收。
