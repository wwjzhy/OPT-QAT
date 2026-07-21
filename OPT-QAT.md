# OPD-QAT 项目任务说明

## 1. 目标

实现 **On-Policy Quantization-Aware Distillation（OPD-QAT）**，研究 INT4 量化模型在自回归生成中的 policy drift，并验证在量化模型自身生成的状态分布上进行蒸馏，是否比固定数据上的 offline QAT/KD 更有效。

核心区别：

- **Offline QAD/QAT**：在训练集已有回答构成的固定上下文上对齐教师；
- **OPD-QAT**：量化学生先采样回答，教师再对学生实际访问的上下文提供 token-level 监督。

ShortOPD 仅作为**训练集和评测集来源参考**，不复用其剪枝方法、蒸馏损失、训练超参数、rollout 长度或 short-to-long controller。

## 2. 集群与项目配置

- 本地仓库：`/Users/wenjun/Project/Low-bit/OPT-QAT`
- 集群仓库：`/work/projects/polyullm/wenjun/Low-bit/OPT-QAT`
- 训练框架：`verl`
- GPU：NVIDIA H800
- 容器：沿用 `verl+cu126+0503`
- 容器镜像：`/lustre/projects/polyullm/container/verl+cu126+0503.sqsh`
- 新 conda 环境：`/lustre/projects/polyullm/wenjun/envs/miniconda3/envs/opt-qat`
- 数据目录：`/work/projects/polyullm/wenjun/Low-bit/OPT-QAT/data`
- 实验输出目录：`/work/projects/polyullm/wenjun/Low-bit/OPT-QAT/outputs`

模型：

- Qwen3-4B：`/work/projects/polyullm/models/Qwen3/Qwen3-4B`
- Qwen3-8B：`/work/projects/polyullm/models/Qwen3/Qwen3-8B`
- 第一阶段关闭 thinking mode。

## 3. OPD-QAT 方法设置

### 3.1 教师与学生

- 教师和学生从同一个 Qwen3 checkpoint 初始化；
- 教师使用 BF16，始终冻结；
- 学生保留 BF16 master weights，在 forward 中执行 INT4 fake quant；
- 学生进行全参数 QAT，不使用 LoRA；
- backward 和 optimizer state 保持高精度。

### 3.2 INT4 fake quant

- 格式：W4A16；
- signed symmetric INT4；
- per-group quantization；
- group size：128；
- 量化所有 Transformer attention/MLP Linear 层；
- embedding 和 `lm_head` 保持 BF16；
- 每次 forward 根据当前权重动态重算 scale；
- 不冻结 scale，不将 scale 设为可学习参数；
- fake quant 使用 Straight-Through Estimator 传递权重梯度。

GPTQ 和 AWQ 仅作为无训练 PTQ baseline，不作为 OPD-QAT 的初始化，也不分别实现 GPTQ-QAT/AWQ-QAT。

### 3.3 On-policy 蒸馏

给定 prompt \(x\)，量化学生通过 sampling 生成：

\[
y_t \sim \pi_Q(\cdot \mid x,y_{<t})
\]

教师在学生状态 \(s_t=(x,y_{<t})\) 上计算分布。第一版使用 teacher-to-student forward KL：

\[
\mathcal L_{\mathrm{OPD}}
=\frac{1}{|M|}
\sum_{t\in M}
\mathrm{KL}\left(
\pi_T(\cdot\mid s_t)
\parallel
\pi_Q(\cdot\mid s_t)
\right)
\]

其中 \(M\) 只包含有效 response tokens。temperature、rollout temperature、最大 response 长度和优化步数均通过配置控制，不从 ShortOPD 继承默认值。

训练闭环：

1. 采样 prompts；
2. INT4 fake-quant 学生生成 rollout；
3. BF16 教师在相同 rollout 上计算 logits；
4. 学生重新 forward 并计算 forward KL；
5. 更新学生全参数；
6. 保存 checkpoint、optimizer、scheduler、随机状态和数据进度。

第一版采用同步的 rollout-train 循环。确认正确后再考虑异步 rollout 或吞吐优化。

## 4. ShortOPD 数据

项目完整复用 ShortOPD 的训练 prompt 构成，共 45,447 条。数据目前尚未下载，需要在 `data/` 中构建并保存来源、版本和去重记录。

### 4.1 训练集

数学，共 14,973 条：

- GSM8K train：7,473；
- MATH train：7,500，与 MATH-500 隔离。

代码，共 15,474 条：

- NVIDIA OpenCodeInstruct：15,000；
  - unit-test signal 为正；
  - 去重；
  - 排除函数名与 HumanEval/MBPP test 重叠的样本；
- MBPP non-test unique tasks：474。

开放指令，共 15,000 条：

- 从 ShareGPT/Vicuna-style conversations 和 UltraChat-200K 采样。

不得将 HumanEval、MBPP test、GSM8K test 或 MATH-500 用于训练。

## 5. ShortOPD 评测集

### 5.1 第一阶段评测

- GSM8K test：1,319 条，exact/normalized answer-match accuracy；
- MATH-500：boxed-answer extraction 与 normalization；
- HumanEval：164 条，execution pass@1；
- sanitized MBPP：257 条，execution pass@1。

代码执行必须放入受限 subprocess sandbox，设置超时和资源限制。

### 5.2 后续开放生成评测

- Alpaca；
- QA；
- Summarization；
- MT-Bench。

ShortOPD 使用固定 EAGLE-style prompt sets，并由 GPT-5.5 按 1–10 分评分。当前阶段暂不实现开放生成 judge；获得原始 prompt/judge 模板并确定 GPT-5.5 API 后再接入。若更换 judge，结果必须标注协议差异，不与 ShortOPD 原表直接比较。

## 6. Baselines

第一阶段：

1. BF16 原模型；
2. INT4 W4A16 fake-quant，无训练；
3. GPTQ W4A16 PTQ；
4. AWQ W4A16 PTQ；
5. Offline QAD：固定训练上下文上的 forward KL；
6. OPD-QAT：学生 on-policy 上下文上的 forward KL。

后续加入：

- LLM-QAT；
- BitDistiller；
- 其他 QAT/KD-based quantization recovery 方法。

Offline QAD 与 OPD-QAT 必须使用相同模型、INT4 配置、训练 prompts、训练 token/step 预算、优化器和评测协议；两者只改变状态分布来源。

## 7. Policy drift 分析

分别在真实/固定上下文和学生 on-policy 上下文上统计：

- teacher/student forward KL；
- top-1 token agreement；
- teacher 对学生生成 token 的 NLL；
- 指标随 token position 的变化；
- 学生与 BF16 教师生成轨迹的首次分叉位置；
- 不同生成长度区间的任务正确率；
- 重复、异常终止和 EOS 比例。

必须区分“单步 logits 误差降低”和“长期生成任务性能恢复”，不能只用训练 loss 支持 policy drift 结论。

## 8. 实现阶段

### 阶段 A：环境与数据

1. 在现有容器内创建 `opt-qat` conda 环境；
2. 安装并验证 verl、PyTorch、Transformers 和评测依赖；
3. 下载并构建 45,447 条 ShortOPD prompts；
4. 构建 GSM8K、MATH-500、HumanEval、sanitized MBPP evaluator；
5. 检查训练/测试泄漏。

### 阶段 B：最小原型

1. 实现 INT4 W4A16 fake quant；
2. 验证量化模块范围、数值范围、scale 更新和 STE 梯度；
3. 实现 offline QAD；
4. 实现同步 OPD-QAT；
5. 用小样本完成 rollout → teacher logits → KL → update；
6. 验证 checkpoint 恢复。

### 阶段 C：Qwen3-4B 实验

1. BF16、fake-quant、GPTQ、AWQ baseline；
2. Offline QAD 与 OPD-QAT 公平对比；
3. 四项第一阶段 benchmark；
4. policy drift 分析；
5. 关键配置消融。

### 阶段 D：扩展

1. Qwen3-8B；
2. LLM-QAT、BitDistiller；
3. 开放生成四项评测；
4. 更多位宽、rollout 策略和长生成实验。

## 9. 验收要求

- 教师无梯度，学生全参数可训练；
- embedding 和 `lm_head` 保持 BF16，其余目标 Linear 层正确执行 INT4 fake quant；
- scale 随当前权重动态更新；
- prompt、padding 和截断 token 不误计入 response KL；
- offline 与 on-policy 管线除状态来源外保持一致；
- 固定 seed 可复现，训练可从 checkpoint 恢复；
- 每次实验保存配置、代码 commit、环境、日志、checkpoint 和逐样本结果；
- 正式集群任务前必须通过小模型/小数据 smoke test；
- “policy drift”“优于 baseline”和“首次”等表述必须由实验或文献证据支持。

## 10. 项目 Skills

1. `cluster-environment`：容器、conda、依赖与 H800 环境检查；
2. `shortopd-data`：下载、过滤、去重和构建训练/评测数据；
3. `int4-fake-quant`：W4A16 fake quant、STE 和数值测试；
4. `opd-training`：verl 中的 rollout、teacher forward、KL 和全参数 QAT；
5. `quantization-baselines`：GPTQ、AWQ、LLM-QAT、BitDistiller；
6. `llm-evaluation`：数学、代码和后续开放生成评测；
7. `policy-drift-analysis`：KL、agreement、分叉位置和长度分析；
8. `experiment-reproducibility`：配置、checkpoint、日志和结果追踪。

具体实现和实验均以本文档为准；遇到模型路径、数据路径、集群资源、算法定义或评测协议不明确时，先向用户确认，不自行假设。
