# Final Project Full File Map Explained

这份说明按“环节/阶段”整理整个 `Final Project/`。目标不是只告诉你“文件在哪里”，而是告诉你：

- 这个文件在整条研究流水线里负责什么；
- 它和前后哪些文件相连；
- 你作为新手，应该先看什么、后看什么；
- 哪些文件是核心，哪些只是中间产物，哪些基本可以忽略。

---

## 1. 顶层总览

项目根目录：

- `Guideline.md`
  - 课程 final project 的正式要求。
  - 你要检查“是否满足课程要求”时，先看它。
- `Proposal.md`
  - 你的项目想法和研究问题定义。
  - 它决定了为什么要做 graph-enhanced latent factor pricing。
- `README.md`
  - 项目总说明，最像“操作手册 + 目录地图”。
- `requirements.txt`
  - 依赖记录，主要用于复现环境。
- `.gitignore`
  - 告诉 Git 哪些自动生成文件不跟踪。

顶层目录：

- `configs/`
  - 每个阶段的参数设置。
- `data/`
  - 原始数据和原始特征文件。
- `scripts/`
  - 命令行入口脚本。
- `src/`
  - 真正的核心源码。
- `outputs/`
  - 运行后生成的中间结果、表格、模型输出、图片。
- `reports/`
  - 项目文字说明、各阶段总结、最终图表。

---

## 2. 你最该先建立的整体理解

这个项目的主流水线可以理解成：

1. 先读课程要求和 proposal。
2. 检查原始数据长什么样。
3. 把原始数据清洗成“月度面板”。
4. 训练无图 benchmark 模型。
5. 构建股票图。
6. 训练图模型。
7. 比较模型效果。
8. 把预测转成投资组合并回测。
9. 做解释性分析和稳健性分析。
10. 生成 final report 用图。

对应最重要的入口脚本顺序是：

- `scripts/check_environment.py`
- `scripts/load_data.py`
- `scripts/inspect_data.py`
- `scripts/build_panel.py`
- `scripts/train_benchmarks.py`
- `scripts/build_graph.py`
- `scripts/train_graph_model.py`
- `scripts/evaluate_model_comparison.py`
- `scripts/backtest_portfolio.py`
- `scripts/run_stage8_analysis.py`
- `scripts/generate_report_figures.py`

---

## 3. 环节 0：课程要求、研究目标、项目协议

### 关联文件

- `Guideline.md`
- `Proposal.md`
- `README.md`
- `configs/main_spec.yaml`
- `reports/project_protocol.md`
- `reports/data_role_mapping.md`

### 每个文件在干什么

- `Guideline.md`
  - 规定课程要求：必须做 OOS、必须有 long-short 和 long-only、要解释 loss function、要做 feature importance。
- `Proposal.md`
  - 说明研究问题：是否把股票关系图加入条件潜在因子模型后，会带来增量价值。
- `README.md`
  - 告诉你整个项目有哪些阶段、运行顺序是什么、每一步产出什么。
- `configs/main_spec.yaml`
  - 把主规格实验的核心研究设定写成参数。
  - 比如：目标变量、OOS 协议、图构建默认方案、评估指标。
- `reports/project_protocol.md`
  - 用研究语言重新描述项目目标和实验原则。
  - 它比 `README.md` 更像“研究设计说明书”。
- `reports/data_role_mapping.md`
  - 解释 `data/` 下每个原始文件理论上应该扮演什么角色。

### 这一环节你应该怎么理解

这是“研究设计层”，还没有真正训练模型，但决定了后面全部代码为什么这样写。  
如果你不先看这层，后面看到模型代码时会只觉得“在跑机器学习”；看了这层后你会知道，项目真正关心的是：

- 用 month `t` 的信息预测 `t+1`；
- 模型要有资产定价解释，而不是纯黑盒预测；
- graph 是为了增强 `conditional beta`，不是为了炫技。

---

## 4. 环节 1：环境、目录、基础工具

### 关联文件

- `requirements.txt`
- `scripts/check_environment.py`
- `scripts/load_data.py`
- `scripts/evaluate.py`
- `scripts/list_benchmarks.py`
- `scripts/train.py`
- `src/data/loaders.py`
- `src/evaluation/metrics.py`
- `src/models/benchmarks.py`
- `src/training/train.py`

### 每个文件在干什么

- `scripts/check_environment.py`
  - 检查当前 Python 环境和依赖是否齐。
- `scripts/load_data.py`
  - 快速浏览 `data/` 目录和 pickle 文件能不能读。
- `scripts/evaluate.py`
  - 打印项目里定义的指标目录，不是主实验评估入口。
- `scripts/list_benchmarks.py`
  - 展示 benchmark 注册表。
- `scripts/train.py`
  - 不是主流水线训练脚本，更像训练工具函数的轻量入口。

- `src/data/loaders.py`
  - 定义项目路径对象；
  - 统一“怎么找到 data/configs/outputs/reports”；
  - 负责列出特征文件、读取 pickle 等。
- `src/evaluation/metrics.py`
  - 定义 Sharpe、OOS R^2、Rank IC、Max Drawdown、pricing regression 等指标。
- `src/models/benchmarks.py`
  - 维护“这个项目有哪些 benchmark”的注册信息。
- `src/training/train.py`
  - 放训练通用工具：随机种子、设备选择、TensorBoard writer、OOS schedule helper。

### 这一环节你应该怎么理解

这是“基础设施层”。  
这些文件本身不直接给你最终结果，但后面的每个阶段都靠它们提供统一的路径、指标、训练设置。

---

## 5. 环节 2：原始数据检查与 Stage 2 数据审计

### 关联文件

- `data/BLACKLIST.pkl`
- `data/UNTRADABLE.pkl`
- `data/csi500_mask_monthly.pkl`
- `data/monthly_returns.pkl`
- `data/mcap.pkl`
- `data/risk_free.csv`
- `data/price.h5`
- `data/FF5.csv`
- `data/HXZ.csv`
- `data/instruction.txt`
- `data/features500/`
- `data/features/`
- `scripts/inspect_data.py`
- `src/data/inspection.py`
- `reports/data_audit_stage2.md`
- `outputs/metadata/data_inspection_stage2.json`

### 每个文件在干什么

- `monthly_returns.pkl`
  - 月度股票收益，是未来要预测的原始来源。
- `risk_free.csv`
  - 日频无风险利率，用来合成下个月的月度 risk-free return。
- `mcap.pkl`
  - 市值，用于 value-weight 组合。
- `BLACKLIST.pkl`
  - 黑名单；某些股票在某些日期不能用。
- `UNTRADABLE.pkl`
  - 不可交易标记。
- `csi500_mask_monthly.pkl`
  - 每个月哪些股票属于中证 500。
- `features500/`
  - 主规格特征宇宙，已经基本对应 CSI 500。
- `features/`
  - 更广的 robustness 特征宇宙。
- `FF5.csv`、`HXZ.csv`
  - 因子基准数据，可用于后续定价比较或背景说明。
- `price.h5`
  - 日频价格库；本项目主流程不是特别依赖它，但保留着。

- `scripts/inspect_data.py`
  - 调用数据检查逻辑，生成审计结果。
- `src/data/inspection.py`
  - 真正读取原始对象并总结 shape、index、columns、时间范围。
- `reports/data_audit_stage2.md`
  - 把 Stage 2 的数据调查写成人能读的报告。
- `outputs/metadata/data_inspection_stage2.json`
  - 机器可读版本的审计结果。

### 这一环节你应该怎么理解

这一步是在回答：

- 这些原始文件到底是什么格式？
- 日期和股票代码怎么对齐？
- `features500` 真的等于 CSI 500 吗？
- `BLACKLIST` 和 `UNTRADABLE` 是日频还是月频？

如果没有这一步，后面的训练很容易“看起来能跑，实际上时间对齐错了”。

---

## 6. 环节 3：构建清洗后的月度面板

### 关联文件

- `configs/cleaning_features500.yaml`
- `scripts/build_panel.py`
- `scripts/run_preprocessing.py`
- `src/data/preprocessing.py`
- `src/features/build_features.py`
- `outputs/panels/main_features500_panel.pkl`
- `outputs/metadata/main_features500_panel_metadata.json`
- `reports/data_audit_stage2.md`

### 每个文件在干什么

- `configs/cleaning_features500.yaml`
  - 指定 Stage 2 面板如何构建：
  - 用 `features500`
  - 是否使用 excess return
  - 是否过滤 blacklist/untradable
  - 是否 winsorize、impute、normalize
- `scripts/build_panel.py`
  - 正式构建面板的入口。
- `scripts/run_preprocessing.py`
  - 更轻量的配置验证入口。
- `src/data/preprocessing.py`
  - Stage 2 核心源码。
  - 最关键函数是 `build_monthly_panel()`。
- `src/features/build_features.py`
  - 维护特征清单/manifest，帮助梳理有哪些特征文件。

- `outputs/panels/main_features500_panel.pkl`
  - 最重要的中间数据文件。
  - 后面 Stage 3、5、7 基本都依赖它。
- `outputs/metadata/main_features500_panel_metadata.json`
  - 记录用了多少特征、多少股票、多少期、过滤掉多少行。

### 这一步到底做了什么

`src/data/preprocessing.py` 做了 5 件核心事：

1. 把 month `t` 的特征和 month `t+1` 的收益对齐。
2. 把日频 risk-free 合成月频，再得到 `target_excess_return`。
3. 把 blacklist/untradable 从日频转成当前月标记并过滤。
4. 对每个月横截面做 winsorize、缺失值填补、标准化。
5. 存成一张“干净的月度股票 × 特征”面板。

### 你要抓住的重点

这张 panel 是整个项目的“主数据底座”。  
后面所有模型，本质上都是在回答：

`给我 month t 的这行特征，我能不能更好地预测 t+1 的收益？`

---

## 7. 环节 4：Stage 3 无图 benchmark 模型

### 关联文件

- `configs/benchmarks_features500.yaml`
- `scripts/train_benchmarks.py`
- `scripts/list_benchmarks.py`
- `src/training/non_graph_benchmark_pipeline.py`
- `src/models/non_graph_benchmarks.py`
- `src/models/torch_models.py`
- `src/models/benchmarks.py`
- `outputs/predictions/stage3_non_graph_predictions.pkl`
- `outputs/latent/stage3_non_graph_exposures.pkl`
- `outputs/latent/stage3_non_graph_factors.pkl`
- `outputs/metrics/stage3_non_graph_metrics.csv`
- `outputs/metadata/stage3_non_graph_run_metadata.json`
- `reports/benchmark_definitions_stage3.md`

### 每个文件在干什么

- `configs/benchmarks_features500.yaml`
  - 定义 Stage 3 跑哪些模型、OOS 参数、神经网络参数。
- `scripts/train_benchmarks.py`
  - Stage 3 主入口。
- `src/training/non_graph_benchmark_pipeline.py`
  - 管理 OOS 切分、循环 refit、保存预测结果。
- `src/models/non_graph_benchmarks.py`
  - 真正定义 3 个 benchmark：
  - `MLPBenchmark`
  - `IPCAStyleBenchmark`
  - `CAEStyleBenchmark`
- `src/models/torch_models.py`
  - 放 MLP、conditional beta MLP 之类的 torch 模型组件。
- `reports/benchmark_definitions_stage3.md`
  - 用文字解释这三个 benchmark 各自代表什么。

### 产出文件怎么理解

- `stage3_non_graph_predictions.pkl`
  - 每个月、每只股票、每个无图模型的预测值。
- `stage3_non_graph_exposures.pkl`
  - 只有 IPCA-style 和 CAE-style 会有 latent exposure。
- `stage3_non_graph_factors.pkl`
  - 保存 latent factor 或 factor mean。
- `stage3_non_graph_metrics.csv`
  - 每个 benchmark 的 OOS R^2、Rank IC、RMSE 等。

### 这一步到底在回答什么

这一步是给 graph 模型树立“公平对手”。  
不是只问“graph 有没有用”，而是问：

- 它能否战胜普通 MLP？
- 能否战胜线性动态 beta 模型 IPCA-style？
- 能否战胜非线性但无图的 CAE-style？

---

## 8. 环节 5：Stage 4 股票图构建

### 关联文件

- `configs/graphs_features500.yaml`
- `scripts/build_graph.py`
- `src/graphs/monthly_graphs.py`
- `src/graphs/build_graph.py`
- `outputs/graphs/features500_similarity_hybrid_manifest.csv`
- `outputs/graphs/features500_similarity_hybrid_stats.csv`
- `outputs/graphs/features500_similarity_hybrid/edges/*.pkl`
- `outputs/graphs/features500_similarity_hybrid/pyg/*.pt`
- `outputs/metadata/stage4_graph_metadata.json`
- `reports/graph_design_stage4.md`

### 每个文件在干什么

- `configs/graphs_features500.yaml`
  - 规定图怎么建：
  - 收益相关性边
  - 特征 cosine kNN 边
  - 特征 euclidean kNN 边
  - 再把它们合并成一张图
- `scripts/build_graph.py`
  - Stage 4 入口。
- `src/graphs/monthly_graphs.py`
  - Stage 4 核心逻辑。
  - 它按月构建动态股票图。
- `src/graphs/build_graph.py`
  - 更底层的图工具，负责转成 `networkx` 或 `PyG Data`。

### 图文件怎么理解

- `...manifest.csv`
  - 总目录，记录每个月的图文件在哪里。
- `...stats.csv`
  - 每个月图的统计量，比如节点数、边数、平均度数。
- `edges/YYYY-MM-DD_edges.pkl`
  - 保存该月的 typed edges 和 combined edges。
- `pyg/YYYY-MM-DD.pt`
  - 保存给 PyTorch Geometric 直接训练用的图对象。

### 你要抓住的重点

这一步不是预测，它是在构造“股票之间的关系上下文”。  
这里的边不是行业边，而是“相似性边”。所以现在的项目结论更准确地说是：

`相似性图是否能增强条件因子暴露估计`

而不是：

`行业图是否有效`

---

## 9. 环节 6：Stage 5 图增强条件潜在因子模型

### 关联文件

- `configs/graph_model_features500.yaml`
- `scripts/train_graph_model.py`
- `src/models/graph_latent_factor.py`
- `src/training/graph_model_pipeline.py`
- `src/models/torch_models.py`
- `outputs/predictions/stage5_graph_predictions.pkl`
- `outputs/latent/stage5_graph_exposures.pkl`
- `outputs/latent/stage5_graph_factors.pkl`
- `outputs/attention/stage5_graph_attention.pkl`
- `outputs/metrics/stage5_graph_metrics.csv`
- `outputs/metadata/stage5_graph_model_metadata.json`
- `reports/graph_model_architecture_stage5.md`

### 每个文件在干什么

- `configs/graph_model_features500.yaml`
  - 定义 Stage 5 模型类型、latent dim、loss 权重、训练 epoch。
- `scripts/train_graph_model.py`
  - Stage 5 入口。
- `src/models/graph_latent_factor.py`
  - 定义图模型本体：
  - `GraphBetaEncoder`
  - `GraphConditionalLatentFactorModel`
- `src/training/graph_model_pipeline.py`
  - 把 panel + graph + OOS blocks 串起来训练。
- `reports/graph_model_architecture_stage5.md`
  - 用研究语言解释 Stage 5 模型在干什么。

### 这个模型和普通 GNN 最大区别

它不是：

- `x, graph -> 直接输出 next-month return`

而是：

- `x, graph -> beta`
- `beta × latent factor -> predicted return`

所以它仍然保留了资产定价/潜在因子框架的解释性。

### 产出文件怎么理解

- `stage5_graph_predictions.pkl`
  - 图模型最终 OOS 预测值。
- `stage5_graph_exposures.pkl`
  - 图模型学到的 beta。
- `stage5_graph_factors.pkl`
  - 图模型训练窗口内的 latent factor embedding。
- `stage5_graph_attention.pkl`
  - 如果用了 GAT，这里会有 attention 信息；默认主结果主要还是 GCN。

---

## 10. 环节 7：Stage 6 模型横向比较

### 关联文件

- `configs/evaluation_features500.yaml`
- `scripts/evaluate_model_comparison.py`
- `src/evaluation/model_comparison.py`
- `src/evaluation/metrics.py`
- `outputs/comparison/stage6_tables/stage6_summary_metrics.csv`
- `outputs/comparison/stage6_tables/stage6_monthly_metrics.csv`
- `outputs/comparison/stage6_tables/stage6_latent_diagnostics.csv`
- `outputs/comparison/stage6_tables/stage6_prediction_correlation.csv`
- `outputs/comparison/stage6_plots/stage6_overall_metric_bars.png`
- `outputs/comparison/stage6_plots/stage6_monthly_rank_ic.png`
- `outputs/comparison/stage6_plots/stage6_monthly_cross_sectional_corr.png`
- `outputs/comparison/stage6_plots/stage6_prediction_correlation_heatmap.png`
- `outputs/metadata/stage6_comparison_metadata.json`
- `reports/stage6_model_comparison.md`

### 每个文件在干什么

- `configs/evaluation_features500.yaml`
  - 指定要比较哪些模型、从哪些 prediction/exposure/factor 文件读数据。
- `scripts/evaluate_model_comparison.py`
  - Stage 6 入口。
- `src/evaluation/model_comparison.py`
  - 对齐共同样本，计算比较表和图。
- `reports/stage6_model_comparison.md`
  - 给出结论性文字解释。

### 这一环节的本质

这是“学术比较表”阶段。  
它回答的是：

- graph 模型在 OOS R^2 上是不是更好？
- 在 Rank IC 上是不是更好？
- 在 pricing error 上是不是更好？

这一步还没有真正变成投资组合，只是从预测/定价角度比较模型。

---

## 11. 环节 8：Stage 7 投资组合回测

### 关联文件

- `configs/portfolio_features500.yaml`
- `scripts/backtest_portfolio.py`
- `src/portfolio/backtest.py`
- `src/evaluation/metrics.py`
- `outputs/portfolio/stage7_weights.pkl`
- `outputs/portfolio/stage7_monthly_returns.pkl`
- `outputs/portfolio/stage7_performance_summary.csv`
- `outputs/portfolio/stage7_signal_coverage.csv`
- `outputs/portfolio/stage7_strategy_coverage.csv`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_equal_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_value_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_short_equal_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_short_value_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_summary_bars_10bps.png`
- `outputs/metadata/stage7_portfolio_metadata.json`
- `reports/stage7_portfolio_results.md`

### 每个文件在干什么

- `configs/portfolio_features500.yaml`
  - 定义组合构建规则：
  - long-short 取前后 10%
  - long-only 取前 10%
  - 同时看 equal/value weight
  - 成本网格 0/10/25 bps
- `scripts/backtest_portfolio.py`
  - Stage 7 入口。
- `src/portfolio/backtest.py`
  - 负责把预测值转成权重，再算组合收益、换手率、Sharpe、回撤。

### 产出文件怎么理解

- `stage7_weights.pkl`
  - 每个月实际买哪些股票、权重多少。
- `stage7_monthly_returns.pkl`
  - 每个月组合收益。
- `stage7_performance_summary.csv`
  - 最重要的组合表现表。
- `stage7_signal_coverage.csv`
  - 看信号覆盖多少股票/多少期。
- `stage7_strategy_coverage.csv`
  - 看每种策略平均持有多少股票等。

### 这一环节的本质

这是“经济价值”阶段。  
课程项目不只关心预测准不准，更关心：

`这些预测能不能真转化成一个像样的 long-short / long-only 组合？`

---

## 12. 环节 9：Stage 8 可解释性与稳健性分析

### 关联文件

- `configs/stage8_features500.yaml`
- `scripts/run_stage8_analysis.py`
- `src/evaluation/stage8_analysis.py`
- `outputs/stage8/tables/stage8_main_results_table.csv`
- `outputs/stage8/tables/stage8_feature_exposure_association.csv`
- `outputs/stage8/tables/stage8_feature_exposure_top_links.csv`
- `outputs/stage8/tables/stage8_permutation_importance.csv`
- `outputs/stage8/tables/stage8_neighbor_monthly_summary.csv`
- `outputs/stage8/tables/stage8_neighbor_edge_mix.csv`
- `outputs/stage8/tables/stage8_graph_robustness_summary.csv`
- `outputs/stage8/tables/stage8_graph_robustness_portfolio.csv`
- `outputs/stage8/tables/stage8_gat_attention_summary.csv`
- `outputs/stage8/plots/stage8_feature_exposure_heatmap.png`
- `outputs/stage8/plots/stage8_permutation_importance.png`
- `outputs/stage8/plots/stage8_neighbor_summary.png`
- `outputs/stage8/plots/stage8_neighbor_edge_mix.png`
- `outputs/stage8/plots/stage8_robustness_prediction_bars.png`
- `outputs/stage8/plots/stage8_robustness_portfolio_bars.png`
- `outputs/stage8/plots/stage8_gat_attention_edge_types.png`
- `outputs/stage8/predictions/*.pkl`
- `outputs/stage8/latent/*.pkl`
- `outputs/stage8/metrics/*.csv`
- `outputs/stage8/metadata/*.json`
- `outputs/stage8/graphs/graph_return_only/...`
- `outputs/stage8/graphs/graph_lookback6/...`
- `reports/stage8_final_summary.md`

### 每个文件在干什么

- `configs/stage8_features500.yaml`
  - 定义 Stage 8 要做哪些解释性和 robustness 变体。
- `scripts/run_stage8_analysis.py`
  - Stage 8 入口。
- `src/evaluation/stage8_analysis.py`
  - 整个项目最复杂的分析文件之一。
  - 它会：
  - 汇总主结果表
  - 做 feature-exposure 关联
  - 做 permutation importance
  - 分析 graph 邻居结构
  - 重新跑 graph robustness 变体
  - 汇总 GAT attention

### Stage 8 下面为什么文件这么多

因为它不只是“读现成结果”，它还会派生很多新分析：

- `predictions/*.pkl`
  - robustness 变体模型的预测。
- `latent/*.pkl`
  - robustness 变体模型的 exposures/factors。
- `metrics/*.csv`
  - robustness 变体的指标。
- `metadata/*.json`
  - robustness 变体的运行元数据。
- `graphs/graph_return_only/...`
  - robustness 里重新构建的图。
- `graphs/graph_lookback6/...`
  - 另一个 robustness 图版本。

### 这一环节的本质

它回答的是：

- graph 模型到底依赖哪些特征？
- top picks 是否处在更稠密的图区域？
- 如果换图、换 latent dim、换成 GAT，结论还成立吗？

这一步让项目从“我有一个结果”升级成“我能解释结果，也能检验结果是否稳健”。

---

## 13. 环节 10：最终报告图生成

### 关联文件

- `configs/report_figures.yaml`
- `scripts/generate_report_figures.py`
- `src/evaluation/report_figures.py`
- `reports/figures/figure_1_sample_coverage.png`
- `reports/figures/figure_1_sample_coverage.pdf`
- `reports/figures/figure_2_graph_overview.png`
- `reports/figures/figure_2_graph_overview.pdf`
- `reports/figures/figure_3_model_comparison.png`
- `reports/figures/figure_3_model_comparison.pdf`
- `reports/figures/figure_4_portfolio_cumulative.png`
- `reports/figures/figure_4_portfolio_cumulative.pdf`
- `reports/figures/figure_5_portfolio_summary.png`
- `reports/figures/figure_5_portfolio_summary.pdf`
- `reports/figures/figure_6_interpretability.png`
- `reports/figures/figure_6_interpretability.pdf`
- `reports/figures/figure_7_graph_robustness.png`
- `reports/figures/figure_7_graph_robustness.pdf`
- `reports/figures/figure_8_gat_attention_exploratory.png`
- `reports/figures/figure_8_gat_attention_exploratory.pdf`
- `reports/figures/figure_manifest.json`
- `reports/figure_guide.md`

### 每个文件在干什么

- `configs/report_figures.yaml`
  - 规定生成报告图时，要从哪些 stage 的表里读数据。
- `scripts/generate_report_figures.py`
  - 你当前打开的文件。
  - 它只是一个很薄的入口，真正工作量不在这里。
- `src/evaluation/report_figures.py`
  - 真正负责生成全部 final report 图。
- `reports/figures/*`
  - 供最终论文/汇报直接使用的图包。
- `reports/figure_guide.md`
  - 告诉你每张图表示什么、来自哪里。
- `reports/figures/figure_manifest.json`
  - 图的机器可读目录。

### 你当前打开文件该怎么理解

`scripts/generate_report_figures.py` 的角色非常简单：

- 读参数；
- 把 config 路径转成绝对路径；
- 调用 `run_report_figure_pipeline()`。

所以如果你想真正理解“为什么会画出 Figure 3、Figure 4、Figure 6”，应该看：

- `src/evaluation/report_figures.py`

而不是只盯着这个脚本本身。

---

## 14. reports/ 目录下所有说明文档应该怎么读

### 建议阅读顺序

- `project_protocol.md`
  - 先建立研究目标。
- `data_role_mapping.md`
  - 再建立原始数据地图。
- `data_audit_stage2.md`
  - 再理解清洗后的数据长什么样。
- `benchmark_definitions_stage3.md`
  - 再理解无图 benchmark。
- `graph_design_stage4.md`
  - 再理解图怎么建。
- `graph_model_architecture_stage5.md`
  - 再理解图模型。
- `stage6_model_comparison.md`
  - 再看预测/定价比较。
- `stage7_portfolio_results.md`
  - 再看经济价值。
- `stage8_final_summary.md`
  - 最后看总结。
- `figure_guide.md`
  - 准备写报告或做汇报时再看。

---

## 15. scripts/ 目录下所有入口脚本总表

- `check_environment.py`
  - 环境检查。
- `load_data.py`
  - 快速查看原始数据。
- `inspect_data.py`
  - 生成 Stage 2 数据审计。
- `run_preprocessing.py`
  - 验证预处理配置。
- `build_panel.py`
  - 构建清洗后的 panel。
- `list_benchmarks.py`
  - 列出 benchmark。
- `train_benchmarks.py`
  - 跑 Stage 3 无图 benchmark。
- `build_graph.py`
  - 跑 Stage 4 图构建。
- `train_graph_model.py`
  - 跑 Stage 5 图模型。
- `evaluate_model_comparison.py`
  - 跑 Stage 6 横向比较。
- `backtest_portfolio.py`
  - 跑 Stage 7 回测。
- `run_stage8_analysis.py`
  - 跑 Stage 8 分析。
- `generate_report_figures.py`
  - 生成最终报告图。
- `evaluate.py`
  - 查看指标目录。
- `train.py`
  - 训练工具函数的演示入口。

可以忽略：

- `scripts/__pycache__/`
  - Python 编译缓存，不需要读。

---

## 16. src/ 目录下所有核心源码总表

### `src/data/`

- `loaders.py`
  - 路径、pickle 加载、特征文件枚举。
- `inspection.py`
  - 原始数据审计。
- `preprocessing.py`
  - Stage 2 面板构建核心。

### `src/features/`

- `build_features.py`
  - 特征清单构建。

### `src/models/`

- `benchmarks.py`
  - benchmark 注册信息。
- `torch_models.py`
  - MLP / conditional beta MLP 等神经网络组件。
- `non_graph_benchmarks.py`
  - MLP / IPCA-style / CAE-style。
- `graph_latent_factor.py`
  - GraphConditionalLatentFactorModel。

### `src/graphs/`

- `build_graph.py`
  - 图结构底层工具。
- `monthly_graphs.py`
  - Stage 4 动态股票图构建主逻辑。

### `src/training/`

- `train.py`
  - 通用训练工具。
- `non_graph_benchmark_pipeline.py`
  - Stage 3 管线。
- `graph_model_pipeline.py`
  - Stage 5 管线。

### `src/evaluation/`

- `metrics.py`
  - 指标函数。
- `model_comparison.py`
  - Stage 6 比较。
- `stage8_analysis.py`
  - Stage 8 解释性与稳健性。
- `report_figures.py`
  - final report 图生成。

### `src/portfolio/`

- `backtest.py`
  - Stage 7 组合与回测核心。

可以忽略：

- 所有 `__init__.py`
  - 主要用于包结构。
- 所有 `__pycache__/`
  - Python 编译缓存，不需要读。

---

## 17. outputs/ 目录怎么读才不会迷路

最简单的读法是按“内容类型”理解，而不是按文件名硬记：

- `outputs/metadata/`
  - 每个阶段的运行说明书。
- `outputs/panels/`
  - 清洗后的主面板。
- `outputs/predictions/`
  - 主实验模型预测值。
- `outputs/latent/`
  - 主实验 latent exposure/factor。
- `outputs/metrics/`
  - 主实验指标表。
- `outputs/graphs/`
  - 主图构建结果。
- `outputs/comparison/`
  - Stage 6 比较结果。
- `outputs/portfolio/`
  - Stage 7 回测结果。
- `outputs/attention/`
  - Stage 5 attention 输出。
- `outputs/stage8/`
  - Stage 8 派生分析和 robustness 全家桶。

### 你最常看的输出文件

- `outputs/panels/main_features500_panel.pkl`
- `outputs/predictions/stage3_non_graph_predictions.pkl`
- `outputs/predictions/stage5_graph_predictions.pkl`
- `outputs/comparison/stage6_tables/stage6_summary_metrics.csv`
- `outputs/portfolio/stage7_performance_summary.csv`
- `outputs/stage8/tables/stage8_main_results_table.csv`

---

## 18. 作为新手，最推荐的“看懂项目”顺序

### 第一轮：只看项目在干什么

- `Guideline.md`
- `Proposal.md`
- `README.md`
- `reports/project_protocol.md`

### 第二轮：只看数据如何变成 panel

- `reports/data_audit_stage2.md`
- `configs/cleaning_features500.yaml`
- `src/data/preprocessing.py`

### 第三轮：只看模型

- `reports/benchmark_definitions_stage3.md`
- `src/models/non_graph_benchmarks.py`
- `reports/graph_model_architecture_stage5.md`
- `src/models/graph_latent_factor.py`

### 第四轮：只看结果

- `reports/stage6_model_comparison.md`
- `reports/stage7_portfolio_results.md`
- `reports/stage8_final_summary.md`
- `reports/figure_guide.md`

### 第五轮：只看你当前这张图怎么来的

- `configs/report_figures.yaml`
- `scripts/generate_report_figures.py`
- `src/evaluation/report_figures.py`

---

## 19. 最后一句总结

如果把这个项目压缩成一句话：

> `Guideline/Proposal` 定义要研究什么，`data + preprocessing` 定义能拿什么数据研究，`Stage 3/4/5` 定义模型怎么建，`Stage 6/7/8` 定义模型好不好、为什么好、稳不稳，`report_figures` 负责把这些结果整理成 final report 可以直接用的图。

这就是整个 `Final Project/` 的完整文件地图。
