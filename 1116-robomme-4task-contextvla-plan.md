# ContextVLA → RoboMME 4task×400 改动方案（2026-11-16 美东，未开工）

fork: github.com/hongzefu/ContextVLA（upstream huiwon-jang/ContextVLA@325d6aa）；本地 /scratch/hongze/ContextVLA-hongzefu，分支 robomme-4task。

## 上游现状（读代码所得）
- Qwen2.5-VL-3B + FAST 动作 token 自回归；初始化 HF `huiwon/ContextVLA-3B-Qwen2.5VL-FAST`（OXE 180k 预训练）。
- 固定 8 帧 × 3 视角全部进 VLM；第 2 层（`internal_projection=2`）把前 7 帧所有视觉 token 取均值成 1 个 context token，之后只保留当前帧 token（`layer_wrapper.py`）。
- 历史帧 = 连续 8 帧（`delta_timestamps` 步长 1/fps），LeRobot 数据加载；不用 proprio state；q01/q99 分位归一化到 [-1,1]。
- 224×224 输入下每图 64 token（8×8），与 4task 每帧 8×8 约定一致。
- 上游缺陷：`contextvla_policy.py` 未 import `modeling_contextvla`；`load_model` 忽略 `ckpt_path`（永远加载 HF 预训练权重）；`num_frames=8`、`num_views=3`、`range(8)`、相机键名硬编码；Evaluation 段 TODO；依赖是 conda+pip。

## 我们的数据（4task-v2-1600ep-604f16da）
- 任务 BinFill / RouteStick / VideoRepick / VideoUnmaskSwap，各 400 集；1,192,918 步、605,611 执行样本。
- 非 LeRobot：`source/data/<i>.pkl` 每个执行样本只含当前帧 front/wrist 256×256、state 8 维（7 关节+夹爪）、actions 20×8、prompt。
- 执行前演示前缀帧（如 ep0 前 678 步）只在原始 h5：本机仅有 `/scratch/hongze/source-4task-h5-604f16da/tar/*.tar.xz`（85G），需解压。/scratch 剩 756G。
- 基线（pi0.5 JAX）：b128、50k、8 卡 FSDP；历史 = 全历史等间隔 8 帧（`even_sampling_indices`）；前 7 维 delta 动作；action_horizon 20、执行 16；评估 692 局 websocket（reset/add_buffer/infer）。

## 改动清单
### A 数据
1. 解压 h5 → 新增 `RoboMMEH5Dataset`（torch Dataset，直接读 h5、在线 resize 224），样本集合 = 同一 605,611 执行样本（DATA_LINEAGE 同源 604f16da）。不转 LeRobot 视频（省时、无压缩损失）。
2. 历史帧索引改为与基线一致的全历史等间隔采样（含演示前缀），替换连续 8 帧。
3. 动作：action_dim=8、horizon=20；前 7 维 delta（复用 transforms.DeltaActions）；q01/q99 取现有 norm_stats.json（delta 口径需重算）。
4. prompt = task_goal 小写。

### B 模型
5. `LayerWrapper` / `process_example` / policy 的帧数、视角数、相机键参数化；视角改 2（front strong aug、wrist weak aug），去掉全零第三视角（省 512 token）。
6. 序列长度：8 帧×2 视角×64 = 1024 视觉 token → 第 2 层后 1+128。

### C 训练
7. 新增 `contextvla_robomme4task` config 与启动脚本；ckpt、日志落 v1-store；依赖改 uv pyproject（torch 2.7.1/cu126、flash_attn 2.8.3、transformers 4.53.2）。
8. 3B 全参 DDP + AdamW 每卡约 60G 优化器状态，可能需梯度检查点或 FSDP，需短测速定 batch。
9. 记录项：loss、grad_norm、lr、动作 token 准确率、FAST 解码失败率、s/步、显存。

### D 评估（692 局）
10. 新 torch policy server，兼容现有 MMEVLA websocket 协议（reset/add_buffer/infer）：按 episode 缓存 front/wrist 全历史（含 add_buffer 推来的执行前帧），同训练口径选帧 → generate → FAST 解码 → 反归一化 → delta+state 还原 → 返回 actions[:16]。
11. 修 `load_model` 读我们 ckpt；FAST 解码失败的回退动作需定义；新 eval server venv。
12. 复用 `local_692_pool.sh` 的分片/每卡 worker 机制，只换 server 启动命令。

## 待裁决
- D1 历史采样：全历史等间隔（推荐，与基线一致）/ 论文连续 8 帧 / 两个都跑。
- D2 帧数口径：8 帧含当前帧（论文）/ 8 历史帧+当前帧（与基线 8 帧记忆对齐）。
- D3 state 是否离散化写进 prompt（pi0-FAST 做法；上游不用）。
- D4 训练预算：对齐基线 b128×50k / 论文 b32×30k–60k；先短测速再定。
- D5 动作：前 7 维 delta（推荐，同基线）/ 绝对。
- D6 h5 解压落点与是否保留解压副本（体量待实测）。
