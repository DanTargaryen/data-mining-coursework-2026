# 数据挖掘导论 2026 · 四项必做与拓展一

**GitHub 代码仓库**：[DanTargaryen/data-mining-coursework-2026](https://github.com/DanTargaryen/data-mining-coursework-2026)

在 Python 3.11.15 / macOS arm64 上验证。原始输入是课程发布的两个图像 ZIP、`weather.csv` 和 `thyroid.zip`。脚本直接读取原始文件。把 `DM_2026_Dataset` 放在代码目录旁边，或将以下命令的 `--data` 改为实际数据目录。

## 任务 1 和 2：图像聚类与异常检测

在 `run.py` 所在的目录运行：

```sh
python3.11 -m venv .venv-images
.venv-images/bin/python -m pip install -r requirements-images.txt
.venv-images/bin/python -u images.py --data ../DM_2026_Dataset --out results/reproduced-images
```

首次运行从 PyTorch 下载 ImageNet ResNet18 权重，缓存到 `.image-cache/`，也可以用 `--cache` 指定位置。实验全程在 CPU 上运行，编码器只做推理。可以用 `--task cluster` 或 `--task anomaly` 单独运行。

- 聚类预设六簇。RGB 直方图与 ResNet18 全局特征各用 KMeans，额外比较固定特征上的 Ward。代码固定分组结果后才读取类别标签，用于 ARI/NMI 和列联表。CSV 保存每张图和各次运行的簇编号。KMeans 用 7、23、42 三个种子，每次 20 个初值。指标 JSON 保留所有原始结果。
- 图像异常检测按榛子和拉链分别建模。各自保留 160 正常和 40 异常用于拟合，40 正常和 10 异常用于验证；按验证 F1 定阈值。对比整图逻辑回归和仅用正常拟合图块建立参考库的局部最近邻。脚本在模型和阈值固定、测试分数产生之后才读测试标签。
- `cluster-assignments.csv` 和 `anomaly-predictions.csv` 保存逐图输出，`anomaly-train-split.csv` 保存拟合与验证划分。两项均生成 PDF/PNG 结果图、未舍入指标 JSON 与权重 SHA-256。模型采用官方预训练权重，未在课程图像上微调。

## 任务 3 和 4：时间序列与表格异常检测

结果目录有逐样本预测、训练/校准清单、指标 JSON、PDF 与 PNG 图。输入为课程发布的 `weather.csv` 和 `thyroid.zip`，脚本只读数据，无需解压 Thyroid 或下载权重。

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --data ../DM_2026_Dataset --out results/reproduced
```

- `--task weather` 和 `--task thyroid` 可分别运行。任务三默认预测 `CO2 (ppm)`。本地课程文件末列 `OT` 自动映射到同一目标。可用 `--weather-csv /path/to/mpi_roof_2020a.csv` 直接读取气象站原始文件。`--target 'T (degC)'` 可运行温度对照，请另设 `--out`。
- 天气按目标时间切分约 70/15/15。全变量岭回归为主实验，使用连续 12 个时刻的全部 21 项指标（252 维）。单变量岭回归只用历史 CO₂，作为对照。标准化仅拟合训练集，两个岭回归的正则各按验证 MAE 确定。图 3 展示全变量模型。预测文件同时保存下一时刻浓度和相对最新观测的变化量，单位均为 ppm。
- 课程文件与[气象站原始 CSV](https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2020a.zip) 的 26,200 行时间戳及 21 列数值按行一致。
- CO₂ 误差差异分析可在任务 3 跑完后复现。若使用上面的时间序列命令，运行 `.venv/bin/python analysis_weather.py --results results/reproduced`。脚本对已有测试预测做一天及两天时间块重抽样，保存 `weather-block-bootstrap.json`。它使用训练段确定的快速变化阈值，不会重新训练或选择模型。
- 甲状腺训练集因相同特征向量分组后才拆为拟合与正常校准。阈值只依赖校准分数的第 95 百分位，官方测试标签只供最终计分。附加分析报告去除训练集同向量测试行后的指标，官方测试集为主结果。
- 表内指标保留三位小数，JSON 保存未舍入值。测试预测文件保留在本地，报告只使用汇总指标与图。

## 拓展一：跨模态异常检测

在 `run.py` 所在的目录使用任务 1、2 的图像实验环境。已有 `.image-cache/` 时可直接复用图块特征。

```sh
.venv-images/bin/python -u extension.py --data ../DM_2026_Dataset --out results/reproduced-extension
```

复用图像与甲状腺的原有划分，只用正常拟合样本建参考库，再由正常校准样本决定告警。统一使用最近平方欧氏距离和分数秩，统一告警规则为 `p <= 0.05`。AUROC 与 AP 使用未离散化的距离分数。三组分开计算指标。

输出 `extension-metrics.json`、`extension-predictions.csv`、`extension-calibration.csv` 和 `extension-ranks.pdf` / `extension-ranks.png`。图像和表格共享评分与校准代码，各组的输入表示与正常参考库单独建立。
