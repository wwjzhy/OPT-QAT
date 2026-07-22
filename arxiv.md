# OPD-QAT 论文写作说明

## 1. 论文定位

- 目标会议：ICLR 2027；
- 当前方法名：**OPD-QAT**，投稿前可修改；
- 文档语言：本说明使用中文，论文正文使用英文；
- 当前版本：按双盲匿名投稿撰写，不出现作者、机构、项目私有地址或可识别身份的信息；
- 论文类型：方法 + 实证分析，不应只写成工程报告。

一句话主张：

> Existing quantization recovery methods align a quantized model with its full-precision teacher on fixed contexts, while deployment exposes the model to contexts induced by its own predictions. OPD-QAT trains on these student-induced states to correct autoregressive policy drift caused or amplified by quantization.

必须谨慎区分：

- 传统 **exposure bias** 是 teacher forcing 训练和 free-running inference 之间的状态分布差异；
- 本文研究的是量化恢复中的类似分布错位：offline QAT/QAD 使用固定上下文，量化模型部署时使用自身生成的上下文；
- 不应在没有实验支持时声称“量化产生了一种全新的 exposure bias”；
- 更准确的表述是 **quantization-amplified exposure bias**、**autoregressive policy drift after quantization** 或 **state-distribution mismatch in quantization recovery**。

## 2. 核心叙事

论文按照以下逻辑展开：

1. 低比特量化降低存储和推理成本，但生成任务上的损伤不能完全由静态 perplexity 或单步 logits 误差解释。
2. LLM 是自回归策略。早期 token 排序变化会改变后续 prefix，使量化模型逐渐进入不同于全精度模型和训练数据的状态分布。
3. 现有 PTQ、QAT 和量化蒸馏通常在真实回答、教师回答或其他固定序列上恢复模型，类似传统 sequence-to-sequence 中的 teacher forcing。
4. 经典 exposure-bias 文献指出：训练时只观察参考 prefix、推理时依赖自身预测，会造成误差累积。
5. 本文首先用位置相关 KL、token agreement、teacher NLL 和轨迹分叉证明量化后的 policy drift 在 student rollout 上更明显。
6. OPD-QAT 让 INT4 学生生成自身轨迹，再由同源 BF16 教师提供 dense token-level forward-KL 监督。
7. 在严格控制量化配置和训练预算后，将 OPD-QAT 与 offline QAD/QAT、PTQ 和现有量化恢复方法比较。
8. 最终结论必须同时由任务性能与 policy-drift 指标支持。

这条叙事的关键不是“on-policy distillation 本身是新的”，而是：

- 明确识别量化恢复中的状态分布错位；
- 系统证明量化误差如何沿自回归轨迹累积；
- 将 on-policy distillation 与 quantization-aware full-parameter training 结合；
- 在相同 INT4 学生和预算下证明 on-policy states 是恢复长期生成行为的关键。

## 3. 与参考工作的关系

### 3.1 Exposure bias

使用以下文献建立传统叙事：

- Teacher forcing：训练时条件于真实历史 token；
- Scheduled Sampling（Bengio et al., 2015）：逐渐用模型预测替代真实历史；
- MIXER（Ranzato et al., 2016）：使用模型生成状态并优化序列级目标；
- Professor Forcing（Lamb et al., 2016）：对齐 teacher-forced 与 free-running dynamics；
- GKD / On-Policy Distillation（Agarwal et al., ICLR 2024）：在学生生成序列上使用教师反馈。

不能只引用经典 RNN 工作后直接宣称其结论适用于量化 LLM。必须解释二者的对应关系：

| Sequence modeling | Quantization recovery |
|---|---|
| Ground-truth/teacher prefix | 固定数据或教师生成 prefix |
| Free-running model prefix | 量化学生自身 prefix |
| Train-test state mismatch | Recovery-deployment state mismatch |
| Compounding prediction errors | Quantization-induced token deviations and policy drift |

### 3.2 BitDistiller

写作上参考 BitDistiller 的清晰结构：

1. 先定义量化与蒸馏问题；
2. 再给出方法算法和训练目标；
3. 实验分为主结果、推理任务、消融和分析；
4. 所有方法组件都有对应消融。

本文与 BitDistiller 的区别必须直接说明：

- BitDistiller 关注 sub-4-bit QAT、自蒸馏、非对称量化/裁剪和 CAKLD；
- BitDistiller 主要在固定序列上蒸馏，并报告 teacher-generated outputs 优于 student-generated outputs；
- OPD-QAT 不把新量化格式或新 KL 作为主要贡献，而是研究**量化学生访问的状态分布**；
- 因此必须在相同实现中比较 ground-truth、teacher-generated 和 student-generated contexts，解释与 BitDistiller 观察不同或一致的条件。

不得模仿 BitDistiller 的文字、图或表述，只参考其论证结构。

### 3.3 三篇直接相关工作

#### Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs

Okoshi et al.（ICLR 2026）研究 2/3-bit reasoning LLM 的 QAT，方法包含：

1. 使用 reasoning + pre-training mixed-domain data 做 block-wise quantization；
2. 使用教师引导的 reward-rectification loss 和 KL 做端到端恢复；
3. 强调 reasoning knowledge 与 pre-training knowledge 对量化和数据分布的敏感性不同。

该方法是本文最接近的 reasoning-QAT baseline，但其恢复阶段是 **off-policy**：论文明确说明不进行 autoregressive rollout，只使用数据集中的固定 sequences。本文需要突出：

- ReasoningQAT 解决 domain coverage 和 reasoning-oriented objective；
- OPD-QAT 解决学生部署轨迹与固定训练轨迹之间的 state-distribution mismatch；
- 两者可能正交，不能把 ReasoningQAT 描述成没有考虑数据分布；
- 在条件允许时，将其 mixed-domain/off-policy objective 作为 baseline，并测试加入 student on-policy states 后是否继续提升。

#### When Reasoning Meets Compression

Zhang et al. 系统比较 quantization、distillation 和 pruning 对 large reasoning models 的影响，并通过 attribution 分析定位敏感权重。其主要发现包括：

- 参数数量对知识记忆的影响可能大于对推理过程的影响；
- distilled reasoning models 的最后一层 `mlp.up_proj` 可能高度关键；
- GPTQ/AWQ 等方法可能过度压缩最后层和部分 `gate_proj`；
- 仅保护约 2% 敏感权重即可显著恢复推理性能。

该工作从**模块/权重敏感性**解释压缩损伤，本文从**自回归状态分布与轨迹漂移**解释损伤。为了证明 OPD-QAT 的收益不是简单修复少数敏感层，实验应增加：

- uniform INT4 与保护最后层/敏感模块的 mixed-precision baseline；
- 不同层的 weight update、activation drift 或 KL 变化；
- 在相同 mixed-precision 配置下比较 offline QAD 与 OPD-QAT。

#### Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery

Xin et al. 提出 QAD：冻结高精度 teacher，使用 KL 将量化 student 对齐到 teacher。该工作表明：

- 对经过 SFT、RL、model merging 的模型，QAD 比复现原 task loss 的 QAT 更稳定；
- QAD 对训练数据领域和质量较鲁棒，部分领域、合成数据甚至错误生成也可提供恢复信号；
- 在其 NVFP4 设置下，简单 teacher-to-student KL 可恢复接近 BF16 的性能。

该工作提供本文最重要的 **offline QAD baseline**。本文不使用 NVFP4，但参考其“高精度 teacher + quantized student + forward KL”框架。核心区别只能是 context source：

- QAD：数据集、SFT 或预生成序列提供固定上下文；
- OPD-QAT：当前量化学生在线生成上下文。

这也构成本文最大的证伪风险：如果 matched offline QAD 已能在 INT4 W4A16 上恢复长期生成，或者 on-policy states 没有额外收益，则不能声称状态分布错位是主要瓶颈。

### 3.4 Related Work 分组

Related Work 只保留与核心论点直接相关的三组：

1. **LLM Quantization and QAT**：GPTQ、AWQ、LLM-QAT、BitDistiller、ReasoningQAT、NVFP4 QAD 等；
2. **Knowledge Distillation for Quantized Models**：logit KD、self-distillation、QAT-based KD；
3. **Exposure Bias and On-Policy Distillation**：Scheduled Sampling、MIXER、Professor Forcing、GKD、ShortOPD。

每组结尾必须明确现有工作缺少什么，不写论文列表式综述。

## 4. 建议标题

暂定标题：

**OPD-QAT: Correcting Autoregressive Policy Drift in Quantized Large Language Models**

备选：

- **Quantization Meets Exposure Bias: On-Policy Distillation for Autoregressive Language Models**
- **Recovering Quantized LLMs on Their Own Generation Trajectories**
- **From Offline Recovery to On-Policy Alignment for Quantized LLMs**

标题中的 “exposure bias” 只有在实验明确支持 train/deployment state mismatch 时使用。

## 5. 论文结构与页数

ICLR 2027 官方规则尚未发布。当前暂按 ICLR 2026 的 9 页主文规划，参考文献和附录不计入主文页数；投稿前必须用 ICLR 2027 官方模板和 Author Guide 重新核查。

建议主文分配：

1. Abstract：约 180–220 words；
2. Introduction：1.2–1.5 页；
3. Related Work：0.7–1.0 页；
4. Preliminaries and Motivation：1.0 页；
5. Method：1.3–1.6 页；
6. Experiments：2.5–3.0 页；
7. Analysis：0.8–1.0 页；
8. Conclusion and Limitations：0.4–0.6 页。

## 6. 各节写作要求

### 6.1 Abstract

使用五段式信息结构，不写背景堆砌：

1. 问题：低比特量化损害自回归生成；
2. 缺口：现有 recovery 在固定上下文上优化；
3. 观察：量化误差在 student-induced states 上累积为 policy drift；
4. 方法：OPD-QAT 在量化学生轨迹上接受 BF16 教师监督；
5. 结果：只填写已验证的模型、位宽、任务和提升。

禁止出现没有结果支持的 “significantly”、“consistently”、“universally” 和 “first”。

### 6.2 Introduction

建议六段：

1. LLM 量化价值及生成能力退化；
2. PTQ/QAT/KD 的现有恢复范式；
3. 用 teacher forcing/exposure bias 解释固定上下文的局限；
4. 给出量化 policy drift 的初步证据，最好引用 Figure 1；
5. 介绍 OPD-QAT 的核心流程；
6. 三条 contributions。

Contribution 最多三条：

1. 量化 policy drift 的问题定义和系统诊断；
2. 基于学生轨迹的 OPD-QAT；
3. 多模型、多任务和受控对比实验。

如果某项实验未完成，不得保留对应 contribution。

### 6.3 Preliminaries and Motivation

依次定义：

- BF16 teacher policy \(\pi_T\)；
- INT4 fake-quant student policy \(\pi_Q\)；
- offline context distribution \(d_{\mathrm{off}}\)；
- student-induced distribution \(d_{\pi_Q}\)；
- token-level divergence；
- sequence-level policy drift。

需要一个受控诊断：

- 在同一批 prompts 上分别构造 fixed/teacher contexts 和 student contexts；
- 比较 BF16 与 INT4 的 KL、agreement 和任务性能随 token position 的变化；
- 证明差异不是仅由回答长度、sampling temperature 或 prompt domain 导致。

不要把参数量化误差直接等同于 exposure bias。应写成量化扰动改变 action distribution，进而改变后续 state visitation。

### 6.4 Method

方法节至少包含：

1. INT4 W4A16 fake quant：symmetric、group size 128、dynamic scale、STE；
2. Student rollout：从 \(\pi_Q\) sampling；
3. Teacher scoring：教师在相同学生 prefix 上计算 logits；
4. Forward-KL objective；
5. 全参数更新和冻结的 BF16 teacher；
6. 算法伪代码与计算成本。

必须明确：

- 只对 response tokens 计算 loss；
- sampling 本身不反向传播；
- 梯度通过学生在固定 sampled prefixes 上的重新 forward 计算；
- embedding 和 `lm_head` 保持 BF16；
- 与 offline QAD 的唯一区别是 context source；
- rollout 与 teacher forward 带来的额外计算和显存成本。

### 6.5 Experiments

围绕四个研究问题组织：

- **RQ1:** Does quantization induce larger divergence on student-generated trajectories than on fixed contexts?
- **RQ2:** Does OPD-QAT outperform offline quantization recovery under matched budgets?
- **RQ3:** Does the advantage grow with generation length and task complexity?
- **RQ4:** Which context source and training component explain the improvement?

主模型：

- Qwen3-4B；
- Qwen3-8B；
- thinking mode 关闭。

主评测：

- GSM8K；
- MATH-500；
- HumanEval；
- sanitized MBPP；
- 后续加入 Alpaca、QA、Summarization、MT-Bench。

Baselines：

- BF16；
- INT4 fake quant without recovery；
- GPTQ；
- AWQ；
- offline QAD；
- LLM-QAT；
- BitDistiller；
- ReasoningQAT 的 off-policy recovery；
- 保护最后层/敏感模块的 mixed-precision quantization；
- OPD-QAT。

公平性要求：

- offline QAD 与 OPD-QAT 使用相同初始化、prompt、量化器、优化器、训练 token/step 预算和 evaluator；
- PTQ 方法与 QAT 方法分组报告，避免混淆；
- 报告训练时间、rollout tokens、teacher tokens 和峰值显存；
- 所有 generation config 完全公开；
- 关键结果至少报告多个 seeds 或置信区间。

### 6.6 Ablations and Analysis

必须优先完成：

1. Ground-truth vs. teacher-generated vs. student-generated contexts；
2. BF16 vs. INT4 在 offline/on-policy states 上的 divergence；
3. 不同生成位置和长度区间；
4. sampling vs. greedy rollout；
5. forward KL 与无 KD 的 QAT；
6. Qwen3-4B 与 Qwen3-8B；
7. uniform INT4 与保护最后层/`mlp.up_proj` 的 mixed precision；
8. 训练收益与 rollout/teacher 计算成本。

可选：

- quantization group size；
- temperature；
- rollout 刷新频率；
- forward/reverse KL；
- 更低位宽。

### 6.7 Limitations

至少讨论：

- 当前主要验证 INT4 W4A16 和 Qwen3；
- fake quant 与真实 kernel 的数值/速度差异；
- on-policy rollout 和 teacher scoring 的训练成本；
- 对 sampling 策略、prompt distribution 和生成长度的依赖；
- policy drift 指标与最终任务质量不一定存在因果关系；
- 是否适用于更低位宽、activation quantization 和其他模型家族尚待验证。

## 7. 必需图表

主文优先保留：

1. **Figure 1：Motivation**
   - fixed contexts 与 student contexts 上的 divergence 随 token position 变化；
   - 用图直接建立 policy-drift 问题。
2. **Figure 2：Method**
   - INT4 student rollout → BF16 teacher scoring → forward KL → student update。
3. **Table 1：Main Results**
   - BF16、PTQ、offline recovery、OPD-QAT。
4. **Table 2：Controlled Context-Source Comparison**
   - ground-truth、teacher-generated、student-generated。
5. **Figure 3：Length and Drift Analysis**
   - 性能或 divergence 随生成长度变化。
6. **Table 3：Efficiency and Ablation**
   - GPU hours、tokens、显存与关键组件。

图表 caption 必须能独立说明模型、位宽、指标和结论。主文不放仅重复正文的图。

## 8. 结果与引用规范

- 实验未完成时使用 `[TBD]`、`[RESULT]`、`[CITATION]`，不得生成看似真实的数值；
- 每个表格数字必须能追溯到 run ID、配置和逐样本输出；
- 引用必须核对原论文，不根据二手摘要编造结论；
- “首次”主张必须完成系统文献检索；
- 相关性不能写成因果；
- 负结果、异常结果和协议差异不能隐藏；
- BitDistiller、GKD、ShortOPD 等方法必须按其真实数据来源和 trajectory source 分类；
- 不得将 teacher-generated sequence KD 错写为 student on-policy KD。

## 9. 英文写作风格

- 每段只表达一个中心论点；
- 先写 claim，再写 evidence 或 explanation；
- 使用短句和明确主语，避免连续名词堆叠；
- 全文统一使用 `full-precision teacher`、`quantized student`、`offline states`、`student-induced states` 和 `policy drift`；
- 方法名首次出现时给全称，之后统一写 OPD-QAT；
- 避免 “obviously”、“clearly”、“dramatically” 等主观词；
- 不把 benchmark 提升描述为模型全面能力提升；
- 不复述表格全部数字，只强调最重要的比较；
- Abstract、Introduction、Conclusion 中的核心结论必须一致。

## 10. ICLR 投稿检查

当前暂按 ICLR 2026 规则准备：

- LaTeX 官方模板；
- 匿名双盲；
- 初投稿主文不超过 9 页；
- references 与 appendix 放在主文之后；
- 包含 reproducibility statement；
- 检查 ethics、limitations 和 broader impact 要求；
- 图表字体、宽度和可读性符合模板；
- 不修改 style file 压缩页面；
- 匿名代码和补充材料不得泄露身份。

ICLR 2027 Author Guide 发布后，必须重新核查：

- abstract/full-paper deadline；
- 页数；
- preprint/arXiv policy；
- author registration 与 OpenReview profile；
- supplementary material；
- reproducibility、ethics 和 LLM-assistance disclosure。

## 11. 初始参考文献

必须至少核查并纳入：

- Bengio et al. (2015), *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*；
- Ranzato et al. (2016), *Sequence Level Training with Recurrent Neural Networks*；
- Lamb et al. (2016), *Professor Forcing*；
- Agarwal et al. (2024), *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*；
- Du et al. (2024), *BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation*；
- Okoshi et al. (2026), *Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs*；
- Zhang et al. (2025/2026), *When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models*；
- Xin et al. (2026), *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*；
- GPTQ；
- AWQ；
- LLM-QAT；
- ShortOPD；
- 与 ultra-low-bit reasoning compression 直接相关的最新工作。

论文写作前建立引用表，记录每篇工作的任务、模型、量化设置、训练状态来源、蒸馏目标和主要结论，避免错误比较。

## 12. 完成标准

论文初稿只有在以下条件满足后才可称为完整：

- Introduction 的每个核心 claim 都有实验或引用支持；
- exposure-bias 类比有清晰定义，不偷换概念；
- offline QAD 与 OPD-QAT 有严格受控比较；
- BitDistiller 的 teacher-generated data 结论得到正面对比；
- 主结果覆盖至少两个模型规模；
- policy drift 有直接测量，而非仅由最终分数推断；
- 所有数字可追溯；
- 局限性和额外训练成本完整披露；
- 使用 ICLR 2027 最终官方模板和规则完成检查。
