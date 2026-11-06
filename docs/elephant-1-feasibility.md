# Elephant-1 在 Ollaya fork 中的可行性与技术方案

状态：设计研究基线，2026-11-06；目标分支：本地 `feat/elephant-1`，基点
`9b8698e39f11a0f0bcd08ad686de51cd1b4f6fa8`。本地 `origin` 是
[`MonadKai/ollaya`](https://github.com/MonadKai/ollaya)，其 GitHub parent 是
[`ollaya-dev/ollaya`](https://github.com/ollaya-dev/ollaya)。用户随后授权自主实现与本机 GPU 验证；
实现及最新验收状态见[Elephant 家族文档](families/elephant.md)。作者模型固定为
[`baselabs/elephant-1@b3f8a93462fb1051a9de031e94a7ff380e254072`](https://huggingface.co/baselabs/elephant-1/tree/b3f8a93462fb1051a9de031e94a7ff380e254072)。
模型的具体行为和参考运行时陷阱见[模型契约](elephant-1-model-contract.md)。

## 结论

**可以在 fork 中做，但不能把现有 Qwen3.5 或通用 GGUF 布局改个名字就作为 Elephant-1 支持。**
最稳妥的实现是新增专用的 `elephant-1-v1` 布局，先做**文本**推理：
构建引用作者原始 safetensors 的 weightless ONNX 图，沿用 Ollaya 的 Rust
请求解析、runner 进程和统一校准/答案层；先做逐问题完整 forward 的正确性基线，
然后在相同 `Engine` 接口背后研究共享前缀缓存。图像能力在获得可信的作者参考
输出后另设门禁。此路线与 Ollaya 的“Python 仅构建时、权重从作者仓库原样下载、
推理进程独立”约束一致，现有 9B Qwen3.5 ONNX 和 Qwen3.5 视觉图可复用。
[仓库模型生命周期](model-lifecycle.md)、[Nimble 9B](families/nimble.md)、
[Decider Vision](families/decider-vision.md)、[runner 接口](../crates/ollaya-runner/src/engine.rs)。

后续实现已经证明**文本推理可用**：完整权重 SHA-256、427 个文本参数加载、
作者参考 goldens、ONNX CPU/CUDA 与原生 daemon API 均通过验证，见
[实际验收](families/elephant.md#full-weight-evidence-y108p62-2026-11-06)。
当前开发 tag 限制为 4096 token；没有进行正式质量/性能发布基准。
作者模型卡宣称 16,384 token 训练上限，
不能把底模 `max_position_embeddings=262144` 当成已验证可用窗口。
[模型卡](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/README.md)、
[模型配置](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/config.json)。

## 从计算契约反推实现

Elephant-1 是 32 层 Qwen3.5 文本骨干（24 层因果 Gated DeltaNet，8 层 full
attention）加视觉塔；后者是否能通过公开自托管路径正确工作仍待确认。full attention
对共享前缀和每个问题后缀分别做**块内双向**注意力，后缀可看前缀，前缀不可看后缀。
答案不是生成文本，而是在答案位置取 256 个单 token 槽位对应的 LM head logits，
按该问题的选项顺序 softmax；本检查点没有额外 readout head，也没有
`calibration.json`（温度 1.0）。因此必须精确复制作者的消息模板、tokenizer、
前后缀切点、块掩码、位置和选项映射。
[作者补丁](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)、
[readout.json](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/readout.json)、
[本地 tokenizer 核查](elephant-1-model-contract.md)。

现有 Ollaya 调用链为 `manifest/decision.json → runner load → Engine::run_json
→ QuestionOutput.logits → daemon 校准与答案`。`Engine` 的接口只需返回每道题按原选项
顺序排列的 logits、token 用量和截断标记；它已经有 `run_images` 扩展点。
实现核查补充：统一答案层和 OneJev 的数值派生约定有差异：Ollaya 四位小数、
score confidence 使用归一化峰值；OneJev 六位小数、score confidence 使用
等级到众数的加权距离。保持 Ollaya 的既有 API 约定，验收应分别记录原始
logits/概率/选中项/期望等级与接口映射，不把此处差异称为模型误差。
新模块应在此 seam 后收纳 Elephant 专有行为，避免修改通用 `llm-logits-v1`、
`jev-style-v1` 或其他家族的含义。
[runner 接口](../crates/ollaya-runner/src/engine.rs)、
[输出结构](../crates/ollaya-runner/src/lib.rs)、
[daemon 与 runner 协议](../crates/ollaya-runner/src/server.rs)、
[现有 Jev-Style 布局](families/jev-style.md)。

建议的模块与文件归属：

| 位置 | 职责及必须验证的接口 |
| --- | --- |
| `convert/ollaya_convert/families/elephant/ref.py`、`layout.py`、`goldens.py` | 固定作者 OneJev + `serve.patch` 的本地绝对路径参考；输出 token、块边界、槽位 raw logits、概率、答案和拒绝案例。参考装载须断言 `attention=noncausal`、`readout=slot_lm`、`calibration=none`。 |
| `convert/ollaya_convert/families/elephant/export.py`、`graphs.py` | 复用 `llm_common/qwen35.py` 的 Gated DeltaNet/Scan 和现有 weightless safetensors 外部数据；仅对 8 层 full attention 加两块掩码，对 linear attention 保持因果；从最后答案位置读槽位 LM head logits。首版不复制模型权重。 |
| `crates/ollaya-decision/src/elephant.rs` | 从原始 `state/questions` 做作者等价的渲染、校验、选项重排和 token 边界定位；不能让通用 parser 的较宽松规则悄悄改变作者拒绝条件。 |
| `crates/ollaya-runner/src/elephant.rs`、`engine.rs` | 加载新布局与 ONNX 图；对每个问题生成完整序列和显式 block/pad mask，返回原选项顺序的 `QuestionOutput.logits`。在接口内统一限制真实可用 token 数、批大小和显存预算；不在 daemon 分散特例。 |
| `convert/ollaya_convert/catalog.py`、`registry/`、`docs/families/elephant.md` | 验证后建立**本地测试 tag**和 manifest，指向作者固定 HF revision 的原始权重、tokenizer；只在 fork 分支/本机测试仓库中保存派生图与记录。正式发布、站点与原仓库均不在本轮范围。 |

首版图可以用 `input_ids [R,T]`、`prefix_lengths [R]`、`valid_lengths [R]`、
`slot_positions [R]` 作为文字输入，输出 `[R,256]` 的 raw slot logits；
`T` 右填充到 Qwen3.5 的 64 token chunk 倍数。构造 full-attention 掩码时，
同一前缀块内和同一后缀块内可双向看见；后缀可看前缀；padding 永远不可作为 key。
这是**待导出验证的接口草案**，不是当前 ONNX 图的既有能力。现有
`Qwen35Trunk` 仅有因果默认 mask，但 `_attention` 已接收显式 bias；
`Decider Vision` 已证明 Qwen3.5 M-RoPE、视觉塔和作者权重字节引用可接入同类图。
[Qwen3.5 export trunk](../convert/ollaya_convert/families/llm_common/qwen35.py)、
[Vision graphs](../convert/ollaya_convert/families/decider_vision/graphs.py)。

## 路线比较

| 路线 | 可复用部分 | 关键代价 / 风险 | 判断 |
| --- | --- | --- | --- |
| 作者 OneJev/PyTorch 作为 Ollaya 常驻后端 | 能尽快得到作者输出；已有文本顺序/批量 fork | 引入 Python、PyTorch、Transformers 作为运行时，绕开 Ollaya 现有 runner 的轻量分发与独立图契约；作者仓库 ID 示例还可能不读取覆盖配置 | **仅作为参考与 goldens 生成器**。 |
| 直接用现有 `llm-logits-v1`、`jev-style-v1` 或通用 Qwen3.5 GGUF | llama.cpp 已有 Qwen3.5、Jev-Style 的非因果块执行和 KV 前缀复制 | 提示模板、两个 block 的边界和 256 槽位 readout 均不同；改变任何一个就改变概率 | **不可直接复用布局**；只复用低层机制。 |
| 专用 llama.cpp + GGUF（包括社区转换） | 当前 libllama runner 已能直接读原始 logits，Jev-Style 展示分块非因果路径；较易获得共享前缀吞吐 | 作者未发布 GGUF；社区量化改变权重且目前公开 Q4_K_M 对 32 题仅 28 题同答案；当前 Ollaya pinned llama.cpp 构建早于新 `/v1/systemone` 扩展。F16 转换、vision projector、超长块不能仅凭可加载判定语义相同；若 tag 指向社区权重，会偏离本仓库“作者权重原样拉取”原则 | **性能备选 / 快速实验**：只在 F16 文本 parity 达标且能说明转换来源、部署方式后考虑；不以社区 Q4 当正确性基线。 |
| 专用 weightless ONNX + 作者原始 safetensors | 9B Qwen3.5 现有 `nimble` 已复用同一 exporter；`decider-vision` 已有图像预处理和 weightless 外部数据；维持 Python 仅构建时 | 需实现双块掩码与专用布局；完整序列每题重算前缀，长序列的计算和显存可能超标，后续缓存图复杂 | **推荐主线**：先证明文本数值，再按测量结果增加缓存；图像另验。 |

社区 GGUF 的验证数据来自其**转换维护者**，不是 Base Labs 的质量保证；
它证明 GGUF 路线值得试验，也说明量化文件不能代替作者 BF16 参考。
[社区 Elephant GGUF 与验证表](https://huggingface.co/waseigo/elephant-1-decision-GGUF)、
[llama.cpp 当前 decision 实现](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-decision.cpp)、
[Ollaya llama.cpp ADR](decisions/0003-llama-cpp-runtime.md)。

## 显存、长度与共享前缀

本机 safetensors 头部显示：语言参数约 15.87 GB、untied LM head 约
2.03 GB、视觉塔约 0.91 GB，合计 18.82 GB，均为 BF16；完整文件仍待哈希校验。
`y108p62` 采样时有 8 张 RTX 5090、每张 32,607 MiB；GPU 4–7 约 32,110 MiB
可用，宿主可用 RAM 约 467 GiB、`/opt` 可用约 5.9 TiB。**这些容量只说明
有做 9B 试验的空间，不证明 16k 长序列或多问题一次执行能放进显存。**
现有 Nimble 9B 图在 24 GB GPU 上采用 4,096 padded-token/session 的预算，
可作为初始保守边界；Elephant 的块内双向注意力还会增加长序列成本。
仅一个 `16 heads × 16384²` 的 full-attention 分数矩阵，用 fp32 就约
16 GiB；与文本骨干及 LM head 约 16.7 GiB 的 BF16 权重同驻留已超过单卡容量。
当前 `Qwen35Trunk._attention` 显式生成该矩阵，因此不能把原样图扩到 16k
并宣称可运行；必须验证 ORT 是否融合成流式/分块 attention，或在后续图中
显式实现内存受控的计算。
[Nimble 9B 的实测与预算](families/nimble.md)、[权重清单](elephant-1-model-contract.md)。

首版完整 forward 是正确性 oracle，可独立核查 block mask 与 raw logits；
但每题都重算 15.87 GB 语言骨干的共享前缀。若基准表明这一成本无法接受，
第二阶段在**同一 `Engine` 接口背后**实现“前缀图 + 后缀图”：显式传递
8 层 full attention 的 K/V 缓存及 24 层 Gated DeltaNet 的卷积/递归状态，
并以作者顺序 fork 输出逐层/逐题比较。现有 ONNX Qwen3.5 exporter 是无状态
`Scan`，这部分是新增研究量，不能称为简单打开缓存。
[Qwen3.5 exporter](../convert/ollaya_convert/families/llm_common/qwen35.py)、
[作者 fork 补丁](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/serve.patch)。

LM head 约 2.03 GB，但输出只需要 256 槽位行（`256×4096×2 ≈ 2.1 MB`
BF16）。应先测量 ORT 是否仍驻留整张 head；若是，可研究让 ONNX external-data
的 256 个小 initializer 分别引用原 safetensors 中对应行的 byte offset。
这能保留“作者权重字节原样引用”，但需实证 ONNX Runtime 对多 offset 的加载、
graph SHA 和 parity；不能把导出的 2.1 MB 行另行当作可再发布权重。
[weightless 导出工具](../convert/ollaya_convert/families/llm_common/onnx_export.py)、
[注册表外部权重层](../crates/ollaya-registry/src/manifest.rs)。

## 最小验证路径与停止条件

1. **参考真值**：等本机模型下载回执 `status=verified`；使用固定 OneJev
   `81ce62f` + 作者 `serve.patch`、**绝对本地模型路径**，检查非因果/slot/无校准，
   且无缺失权重键。若作者文本参考本身无法正确加载，停止推理适配并先确认上游。
2. **布局**：先做不需要 GPU 的 Python/Rust token 序列 parity；覆盖 `noul`、
   `choice`、`score`、1/26/27/255 选项、JSON state、非 ASCII、多问题、特殊 token、
   16,384 token 边界和拒绝案例；逐项比较 prefix 长度、slot token、选项映射。
3. **计算**：先导出小输入 eager 图，对作者完整 forward 与顺序 fork 取 raw slot
   logits；再测 ONNX CPU/CUDA、掩码、padding 和模型答案。把 BF16 作者参考、
   fp32 计算和可能的 fp16 tag 分别记录；不能把量化结果混成同一验收。
   失败应定位数值源，不放宽容差或删除案例。
4. **资源**：在明确空闲的本机 5090 上测 512/2048/4096/8192/16384 token 与
   1/4/16 问题的峰值显存、延迟和 OOM；若仅短上下文通过，就在 tag 中公开真实
   可用上限，超限返回 422，不截断或悄然改用因果 mask。共享前缀图只有在
   短文本 parity 已经稳定后进入。
5. **图像**：先获得正确的作者图像 goldens；再复用现有 `vision.rs` /
   `decider_vision` 的预处理、M-RoPE 与两图结构，验证图像顺序及共享前缀。
   公开自托管补丁没有把非因果覆盖接到 `MMDecisionEngine`，在此之前 tag 应
   拒绝图像输入并报告“text only”，不能只因权重含视觉塔就宣称视觉支持。
6. **本地交付**：使用 fork 本地 `feat/elephant-1` 分支、`convert` 的模型家族、
   parity 回执和本地测试 registry；`ollaya pull` 的 manifest 指向 Base Labs 固定
   HF 原始权重。按 [模型生命周期](model-lifecycle.md)完成相关 K1/K2/K3 门禁。
   不向 `ollaya-dev/ollaya` 推送、不建上游 PR，不发布模型权重。

**需要明确的首个试验结果**：作者指定环境在本地固定快照上能否以
`noncausal` + `slot_lm` 完整加载、并输出可复现的文本 raw logits。
这项检查现已通过：固定作者参考成功加载全部文本参数，完整模型输出已成为
12 行 CPU/CUDA parity 的真值；图像和更长上下文仍另行验收。
