# GitHub更新流程

代码和实验配置存入Git。data/、result/、runs/、wind-env/、缓存、模型检查点和Slurm日志由.gitignore排除。logs/README.md保留日志约定，reports/保存小型分析快照。AGENTS.md沿用用户已有的本地忽略设置。

## 2026-11-22：本轮上传范围

当前分支main，origin指向https://github.com/LiangTian49/wind-prediction.git。尚未替用户暂存、commit或push，也未核实远端最新提交或仓库可见性。

| 内容 | 本轮建议 | 原因 |
|---|---|---|
| 根目录Python、configs/、slurm/、tests/、scripts/ | 上传 | 包括已完成实验与新年下载/恢复/确认入口；实现不等于真实执行完成 |
| README、CURRENT_STATUS、操作指南与实验记录 | 上传并注明状态 | 当前下载3/24成功；恢复与新年确认仍待验收，不写成已完成 |
| environment/依赖清单、约束、激活脚本 | 上传 | 包括新增CDS依赖；不上传venv和安装缓存 |
| reports/小型汇总表、来源manifest与选定图件 | 上传 | 已从归档结果复制，不重新计算实验 |
| legacy/及有迁移记录的docs/archive/ | 一并保留 | Git显示旧路径删除、新路径新增；原文件并未丢失 |
| main.tex、figures/、docs/paper/ | 可选，默认本轮暂不上传 | 包含未投稿稿件、原稿备份和重复图片；私有仓库可按合作约定同步，公开前决定分享范围 |
| 合作者反馈、内部任务书原文 | 默认本轮暂不上传 | 与对外可公开的方法/结果文档分开选择 |
| data/、runs/、result/、model/检查点、wind-env/ | 不上传Git | 原始GRIB/遥感栅格、逐样本预测、模型/预处理器和完整环境保留在集群 |
| logs/日志、CDS key、缓存、LaTeX build/.texcache、上传ZIP | 不上传 | 日志可能含临时服务信息；凭据不进入仓库；编译产物可重建 |

.gitignore已经补齐论文build、.texcache、upload_bundle.zip和LaTeX辅助文件规则。没有删除本地文件；也没有自动屏蔽论文源码，是否上传由用户决定。不要用git add -f绕过数据/模型/凭据的忽略规则。

10个已跟踪旧路径目前显示D，已逐一用HEAD内容核对迁移后文件，字节完全一致：6个脚本到legacy/、note.txt到docs/archive/、3张旧图到docs/archive/figures/。应同时提交新旧路径，Git通常会识别为重命名；无需为了消除D把文件又复制回根目录。

## 推荐操作：先更新代码与结果，保留论文及反馈本地

从项目根目录执行：

```bash
source environment/activate_wind_env.sh
python scripts/export_reports.py
git status --short
git diff --check

git add -A -- . \
  ':(exclude)main.tex' \
  ':(exclude)figures/**' \
  ':(exclude)docs/paper/**' \
  ':(exclude)docs/experiments/WIND_FEEDBACK_PLAN_UPDATE_2026-11-11.md' \
  ':(exclude)docs/experiments/WIND_EXPERIMENT_TASKS_FOR_CODEX.md'

git diff --cached --stat
git diff --cached --name-status
git diff --cached --check
```

上面的范围已做dry-run验证。它暂存代码、配置、维护说明、汇总结果和目录迁移，排除论文目录/根稿/根图件及两份内部反馈文件。排除项只是本轮不暂存，未删除本地文件。如果这些材料先前已被手动暂存，exclude不会撤销旧暂存，需要先用git restore --staged对具体路径取消暂存。

检查清单无误后，再执行：

```bash
git commit -m "Organize CERRA experiments and add independent-year download recovery"
git fetch origin
git log --oneline --left-right HEAD...origin/main
git push origin main
```

若远端main有他人新提交，先整合再push，不用force覆盖。此处没有自动fetch、提交或推送。第一次SSH/HTTPS认证问题按GitHub账户既有方式处理，CDS token不能用于GitHub认证。

## 可选：同步论文材料

若仓库为私有、且你决定同步论文协作资料，可以另作一次提交：

```bash
git add main.tex figures docs/paper
git diff --cached --stat
git diff --cached --name-only
```

审阅后再commit/push。docs/paper中含多份稿件版本、审阅记录及原稿PDF；不是仅当前main.tex。选择性分享时只添加所需具体路径。编译build、缓存和upload_bundle.zip仍被忽略。保留PDF预览与精选PDF/PNG图件是可选项，两个图片目录包含重复副本，不必为了代码复现都上传。

若本轮不上传论文/反馈，README或文档中指向这些本地材料的链接在GitHub上不会可用；公开整理时可按实际分享范围移除相应导航，不应把未上传的稿件误称为公开材料。

## 日常更新

阶段验收后更新CURRENT_STATUS、PROGRESS_LOG及实验总记录，运行小型快照导出，再按所选分享范围暂存、审阅、commit与push。不要将提交过的作业写成已完成，也不要把远端请求排队写成数据已经下载。Git只保存版本记录，不替代data/runs/result的集群备份。

新增Slurm作业放slurm/，继续从项目根目录提交，日志使用项目logs/绝对路径。后续新增Python模块可以在经过协议回归检查后逐步分包，目前生产模块保持原路径以保留冻结摘要和结果追溯。

## 环境与可复现记录

environment/requirements-wind-env.txt记录整理时的依赖快照，requirements-cds-download.txt记录安装作业913326新增的4个下载包；原requirements.txt保留。保存实验配置、数据版本和代码摘要继续由训练入口负责。Slurm脚本内的JSC绝对路径根据实际集群工作目录配置。上传依赖列表即可，不上传wind-env目录。
