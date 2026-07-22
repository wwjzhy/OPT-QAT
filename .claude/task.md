## 2026-07-21 18:14 - 编写 OPD-QAT 项目任务说明

### 目标
- 将论文构想整理为可执行、可验证的代码与实验任务说明，为后续项目 skills 和实验实现提供统一依据。

### 修改
- `OPT-QAT.md`：新增研究问题、方法定义、实验范围、baseline、公平性约束、代码模块、阶段验收、测试、实验记录和建议 skills 划分。
- `.claude/task.md`：记录本次文档任务。

### 关键决策
- 沿用论文方法名 `OPD-QAT`，文件名按用户要求使用 `OPT-QAT.md`。
- 将“策略漂移”“优于现有方法”和“首次”等表述定义为待验证假设，避免在文献核查和实验完成前作为结论。
- 第一版优先实现同步的 rollout、教师打分和 LoRA 更新闭环，再考虑异步或分布式性能优化。

### 难点与不确定项
- ShortOPD 的具体训练集、测试集、长度课程和评测器尚未核查。
- LLM-QAT、BitDistiller 与 GPTQ/AWQ 是否能在完全一致的量化格式和预算下公平比较，仍需依据公开实现确认。
- 最终 benchmark、位宽组合、训练预算和集群资源尚待确定。

### 验证
- 已检查任务说明包含目标、实现、实验、测试、复现和完成标准。
- 本次仅新增 Markdown 文档，未运行代码测试。

### 后续
- 按 `OPT-QAT.md` 第 10 节逐个创建项目级 skills，优先建议从 `literature-audit` 和 `on-policy-rollout` 开始。

## 2026-07-21 18:33 - 精简任务说明并补充 ShortOPD 设置

### 目标
- 删除重复的流程性说明，将 ShortOPD 的数据、训练和评测细节直接写入任务文档。

### 修改
- `OPT-QAT.md`：由 12 个宽泛章节精简为方法、ShortOPD 设置、本项目实验、实现顺序和验收要求；补充 ShortOPD 的 45,447 条训练 prompts、八类 benchmark、训练超参数和 short-to-long controller 参数。

### 关键决策
- ShortOPD 细节以 arXiv:2607.13124v1 原文为准。
- 第一阶段先实现固定 2,048-token OPD-QAT，确认训练闭环后再接入 repetition-gated controller。
- 保留与实验执行直接相关的复现和公平性要求，删除通用代码目录建议及重复工作原则。

### 难点与不确定项
- ShortOPD 开放生成评测使用 GPT-5.5 judge；若无法获得相同 judge，替代结果不能与论文原表直接比较。
- Qwen3-8B、GPTQ/AWQ 和更低位宽属于本项目扩展，不是 ShortOPD 原论文设置。

### 验证
- 已对照 ShortOPD 原文的 Setup、Training details 和 Evaluation details。
- Markdown 文件未发现代码级测试需求。

### 后续
- 按精简后的实现顺序创建数据评测、量化 baseline 和 OPD 训练 skills。

## 2026-07-21 18:41 - 按确认配置重写任务说明

### 目标
- 纠正对 ShortOPD 的过度引用，并将用户确认的集群、模型、INT4 QAT 和评测配置写入任务说明。

### 修改
- `OPT-QAT.md`：删除 ShortOPD 的剪枝、JSD、训练超参数、rollout budget 和 controller 设置；仅保留其 45,447 条训练 prompts 与八类评测集。
- `OPT-QAT.md`：补充 verl、H800、容器、conda、集群代码/数据/模型路径。
- `OPT-QAT.md`：明确 INT4 W4A16、group size 128、symmetric per-group、动态 scale、全参数 QAT、forward KL 和 sampling rollout。
- `OPT-QAT.md`：将 GPTQ/AWQ 限定为 PTQ baseline，并将开放生成 judge 延后。

### 关键决策
- OPD-QAT 使用 BF16 teacher 与同源 INT4 fake-quant student，embedding/lm_head 保持 BF16。
- scale 每次 forward 根据当前权重重算，不冻结且不可学习。
- 第一阶段关闭 Qwen3 thinking mode，只实现数学和代码四项评测及最小原型。

### 难点与不确定项
- ShortOPD 的 EAGLE-style 开放生成 prompt/judge 模板仍需从其公开代码或数据中获取。
- rollout temperature、最大生成长度和训练超参数保留为可配置项，尚未固定。
- verl 与全参数 INT4 fake quant 的具体接入方式需在原型阶段验证。

### 验证
- 已核对文档中的用户确认项与 ShortOPD 原文的数据和评测定义。
- 已确认 H800 不需要也不采用 NVFP4/Model Optimizer 路径。

### 后续
- 创建集群环境与数据构建 skills，然后实现最小可运行原型。

## 2026-07-22 13:14 - 编写 ICLR 2027 论文写作说明

### 目标
- 为 OPD-QAT 建立中文论文写作规范和英文论文逐节提纲，围绕 exposure bias、teacher forcing 和 BitDistiller 组织论文叙事。

### 修改
- `arxiv.md`：新增论文定位、核心叙事、参考工作关系、章节提纲、实验问题、图表要求、引用规范、ICLR 投稿检查和完成标准。

### 关键决策
- 暂用 `OPD-QAT`，论文按匿名 ICLR 投稿版设计。
- 将本文问题表述为量化恢复中的 state-distribution mismatch 或 quantization-amplified exposure bias，避免直接把量化误差等同于经典 exposure bias。
- BitDistiller 作为量化蒸馏结构参考和必须正面对比的 offline baseline；其 teacher-generated data 结论不能直接用于支持 OPD-QAT。
- ICLR 2027 规则尚未发布，暂按 ICLR 2026 的 9 页匿名稿要求规划，并要求投稿前重新核查。

### 难点与不确定项
- 最终方法名、标题和 ICLR 2027 官方格式尚未确定。
- ShortOPD 开放生成 prompt/judge 模板与 GPT-5.5 评测仍待获取。
- 论文中的性能主张和数值必须等待实验完成。

### 验证
- 已核查 Scheduled Sampling、MIXER、Professor Forcing、GKD 和 BitDistiller 的核心论点。
- 已检查文档未把未完成实验写成既定结果。

### 后续
- 建立详细 related-work 引用表，并在实验完成后依据 run 结果填充英文论文初稿。

## 2026-07-22 13:30 - 补充量化推理相关工作

### 目标
- 将 ReasoningQAT、reasoning compression 分析和 NVFP4 QAD 纳入论文定位，并明确它们对 OPD-QAT 新颖性与实验设计的影响。

### 修改
- `arxiv.md`：新增三篇直接相关工作的核心方法、与 OPD-QAT 的区别和必须增加的对照实验。
- `arxiv.md`：在 baseline、消融和初始参考文献中加入 ReasoningQAT、sensitive-layer mixed precision 和 NVFP4 QAD。

### 关键决策
- 将 ReasoningQAT 视为最接近的 off-policy reasoning-QAT baseline，而非忽略数据分布的工作。
- 将 NVFP4 QAD 视为最重要的 matched offline-QAD baseline，并明确其可能证伪 OPD-QAT 核心假设。
- 增加敏感层保护对照，以排除收益仅来自最后层或 `mlp.up_proj` 修复的解释。

### 难点与不确定项
- ReasoningQAT 的完整复现成本和与当前 INT4 W4A16 设置的可比性需进一步核查。
- `When Reasoning Meets Compression` 的敏感层结论主要来自 reasoning/distilled model，是否适用于当前非 thinking Qwen3 需要实验验证。

### 验证
- 已核查三篇论文的原始摘要、方法描述及主要结论。
- 文档未将不同量化格式下的结论直接视为可横向比较结果。

### 后续
- 先做 matched offline QAD 与 OPD-QAT 小规模对照，判断论文核心假设是否成立。
