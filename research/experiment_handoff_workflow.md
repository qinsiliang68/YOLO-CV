# Experiment Handoff Workflow

本仓库后续采用“双线”工作方式：

- `main`：主线，仅用于论文、代码、配置、筛选后的实验材料。
- `exp-dropoff`：投递线，仅用于训练机上传原始实验结果。

这样做的原因很简单：

- 训练机只负责跑实验和上传结果。
- 本地工作机负责拉取结果、筛选可用材料、更新论文、删除主线里不需要的内容。
- 训练机既然“不拉代码、不拉论文”，就不要让它和 `main` 直接互相覆盖。

## 角色分工

### 训练机

训练机只做三件事：

1. 跑实验
2. 把结果整理到 `research/materials/` 或 `research/results/`
3. 推送到 `exp-dropoff`

训练机不要做：

- 不修改 `essay/`
- 不修改主线说明文档
- 不删除主线已有材料
- 不向 `main` 直接推送

### 本地工作机

本地工作机负责：

1. `fetch` / `pull` `exp-dropoff`
2. 从投递结果中挑选要保留的文件
3. 删除主线不需要的材料
4. 更新论文、表格和分析文字
5. 推送到 `main`

## 目录约定

训练机上传时，只使用下面两个目录：

- `research/materials/`
- `research/results/`

推荐规则：

- 每个实验一个独立子目录
- 目录名统一包含模型、任务、数据集
- 不上传原始数据集
- 不上传完整 `runs/` 大目录
- 不上传大权重文件

## 训练机命令

第一次切到投递线：

```powershell
git fetch origin
git checkout -b exp-dropoff origin/exp-dropoff
```

之后每次实验完成后：

```powershell
git add research/materials research/results
git commit -m "Add experiment materials"
git push origin exp-dropoff
```

## 本地工作机命令

拉取训练机最新投递结果：

```powershell
git fetch origin
git checkout main
git pull origin main
git checkout origin/exp-dropoff -- research/materials research/results
```

然后：

- 保留需要的材料
- 删除主线里不需要的旧材料
- 更新论文
- 最后提交并推送：

```powershell
git add research essay
git commit -m "Update thesis with curated experiment results"
git push origin main
```

## 重要约束

- `exp-dropoff` 视为“投递箱”，本地工作机原则上不回写、不改历史。
- 如果训练机永远不拉代码，就不要在 `exp-dropoff` 上做 rebase、回滚或清理。
- 真正的清理工作在 `main` 上进行。

## 当前建议

后续实验都按下面方式理解：

- 训练机：只推 `exp-dropoff`
- 本地工作机：只整理 `main`

这样最不容易冲突，也最符合当前两台机器的使用方式。
