# 美学模型接入说明与 TODO

本文档用于后续扩展美学模型评估，当前已完成 ArtiMuse 与 ArtQuant 框架接入，UNIAA 暂缓处理。

## 一、ArtiMuse（先实现）

- 仓库：/home/Hu_xuanwei/ArtiMuse
- 模型路径：/home/Hu_xuanwei/model/ArtiMuse
- 建议环境：artimuse（独立）
- 依赖参考：ArtiMuse/requirements.txt
- 已在框架中新增配置：configs/models/artimuse.yaml
- 运行脚本：scripts/run_infer_artimuse.sh

加载说明：
- ArtiMuse 为 InternVL 系架构，可先复用 internvl 适配层。
- 如果后续需要其专用 score() 或多维属性输出，再新增专用 adapter。

## 二、UNIAA（先实现）

- 仓库：/home/Hu_xuanwei/Uniaa
- 基准脚本：Uniaa/UNIAA-Bench/aesthetic_assessment.py 与 aesthetic_describe.py
- 模型路径：/home/Hu_xuanwei/data/UNIAA/model
- 建议环境：uniaa（独立）
- 已在框架中新增配置：configs/models/uniaa.yaml
- 运行脚本：scripts/run_infer_uniaa.sh

加载说明：
- UNIAA-LLaVA 可先用 llava 适配层接入描述任务。
- 若后续做 logits 评分（good/poor 档位），建议新增 assessment task 插件。

## 三、ArtQuant（已接入）

- 仓库：/home/Hu_xuanwei/ArtQuant
- 模型路径：/home/Hu_xuanwei/model/ArtQuant
- Base 模型：/home/Hu_xuanwei/model/mplug-owl2-llama2-7b
- 额外关系：ArtQuant 需要基于 mplug-owl2-llama2-7b，再加载 APDD/ArtQuant 相关权重。

已完成：
1. 在 configs/models 中新增 artquant.yaml。
2. 在 adapters 中新增 artquant adapter（src/aesthetic_eval/adapters/artquant.py）。
3. 新增运行脚本 scripts/run_infer_artquant.sh。
4. 输出已对齐统一协议（prediction/reference/image_resolved）。
5. 推理默认改为生成式描述模式（infer_mode: generate），使用样本问题而非固定评分问句。
6. 小样本 smoke 已跑通，输出为自然语言描述。

待验证：
1. 如需复现实验中的离散等级评估，可切换 infer_mode: score，并验证标签到分数映射策略（excellent/good/fair/poor/bad）。

建议环境命令：
1. conda create -n artquant python=3.10 -y
2. conda activate artquant
3. cd /home/Hu_xuanwei/ArtQuant && pip install -e .
4. cd /home/Hu_xuanwei/aesthetic_eval_framework && bash scripts/run_infer_artquant.sh

## 四、AesExpert（已接入）

- 仓库：/home/Hu_xuanwei/AesExpert
- 模型路径： /home/Hu_xuanwei/model/AesMMIT_LLaVA_v1.5_7b_240325
- 环境： aesexpert
- 状态：已接入（aesexpert 专用 adapter）

已完成：
1. 在 configs/models 中新增 aesexpert.yaml。
2. 新增 aesexpert adapter（src/aesthetic_eval/adapters/aesexpert.py），走 LLaVA 官方加载链。
3. 新增运行脚本 scripts/run_infer_aesexpert.sh。
4. 输出协议与其他模型保持一致（prediction/reference/image_resolved）。

待验证：
1. 在 aesexpert 环境执行端到端实跑，确认本地权重目录可直接加载。

## 五、OneAlign（已接入）

- 仓库：/home/Hu_xuanwei/Q-Align
- 模型路径： /home/Hu_xuanwei/model/one-align
- 环境： onealign
- 状态：已接入（新增 onealign adapter）

已完成：
1. 在 configs/models 中新增 onealign.yaml。
2. 在 adapters 中新增 onealign adapter（src/aesthetic_eval/adapters/onealign.py）。
3. 新增运行脚本 scripts/run_infer_onealign.sh。
4. OneAlign 输出连续美学分数，并写入统一 prediction 字段。

待验证：
1. 在 onealign 环境执行端到端实跑，确认 q_align 依赖链稳定。

## 六、Q-SIT（已接入，待环境）

- 仓库：/home/Hu_xuanwei/Q-SiT
- 模型路径： /home/Hu_xuanwei/model/q-sit
- 状态：已完成框架接入，待可用环境

已完成：
1. 在 configs/models 中新增 qsit.yaml。
2. 在 adapters 中新增 qsit adapter（src/aesthetic_eval/adapters/qsit.py）。
3. 新增运行脚本 scripts/run_infer_qsit.sh。

待验证：
1. 现有 onealign/artimuse 环境的 transformers 版本不支持 LlavaOnevisionForConditionalGeneration。
2. 需新建 qsit 环境（建议 transformers>=4.45）后再做端到端实跑。

## 七、UniPercept（已接入）

- 仓库：/home/Hu_xuanwei/UniPercept
- 模型路径： /home/Hu_xuanwei/model/UniPercept
- 状态：已接入（复用 artimuse 环境）

已完成：
1. 在 configs/models 中新增 unipercept.yaml。
2. 在 adapters 中新增 unipercept adapter（src/aesthetic_eval/adapters/unipercept.py）。
3. 新增运行脚本 scripts/run_infer_unipercept.sh。
4. 在 artimuse 环境完成端到端 smoke 实跑，输出目录：outputs/unipercept_description_20260404_183354。
5. 兼容了两类 UniPercept 权重接口差异（有/无 score 方法、chat 参数签名不同），默认 reward 模式可稳定返回 iaa/iqa/ista。
6. 已将默认推理模式切换为 generate，并复用官方 VQA 的 chat 调用路径（按样本 question 生成文本）。

待验证：
1. 在更大样本上评估 generate 模式的文本质量和稳定性。
2. 如后续需要回归评分任务，可切换 infer_mode: reward，保留 score/chat 回退链路。

## 八、环境管理建议

由于内存和磁盘限制，建议每个模型保持独立环境，并复用 env_snapshots 流程：

1. 训练/推理前恢复环境。
2. 跑完立即导出快照并删除环境。
3. 统一在 README 的 Environment Snapshot And Reuse 节操作。
