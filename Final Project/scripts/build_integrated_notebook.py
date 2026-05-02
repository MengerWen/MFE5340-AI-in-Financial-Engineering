"""Build the integrated Chinese showcase/reproducibility notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "Final_Project_Integrated_Showcase.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(md(
        r"""
        # 图结构增强的条件潜在因子定价：完整整合展示 Notebook

        ## 0. Project Roadmap / 项目路线图

        这个 notebook 把原本分散在 `src/`、`scripts/`、`configs/`、`outputs/` 和 `reports/` 中的 final project 整合成一条适合向老师展示的主线。

        **核心研究问题：**  
        在中国股票横截面中，把股票之间的关系结构显式放进条件潜在因子模型，是否能更好地学习 time-varying beta / conditional exposure，并进一步改善样本外定价、收益排序和组合表现？

        **一句话结论：**  
        原始动态图模型没有在所有统计指标上全面胜出，但它在 Rank IC、截面相关性和 long-only 经济价值上展现了清晰增量；新增行业分类扩展说明，静态行业关系确实有定价/拟合信息，但不能替代动态相似性图的排序与组合优势。

        **阅读方式：**
        1. 默认执行只读取已保存结果，不会重跑训练，也不会覆盖原输出。
        2. 每个阶段都包含：研究动机、关键代码、配置、结果表、图和解释。
        3. 最后一节提供完整重跑命令，只有手动设置 `RUN_FULL_PIPELINE = True` 才会执行。
        """
    ))

    cells.append(code(
        r"""
        # 0. 全局控制与辅助函数
        from __future__ import annotations

        import json
        import subprocess
        import sys
        from pathlib import Path
        from textwrap import dedent

        import numpy as np
        import pandas as pd
        from IPython.display import HTML, Image, Markdown, display

        # 如果从 Final Project 目录打开 notebook，PROJECT_ROOT 就是当前目录；
        # 如果从仓库根目录打开，也会自动定位到 Final Project。
        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name != "Final Project" and (PROJECT_ROOT / "Final Project").exists():
            PROJECT_ROOT = PROJECT_ROOT / "Final Project"

        RUN_FULL_PIPELINE = False
        SHOW_LONG_CODE = True

        pd.set_option("display.max_columns", 80)
        pd.set_option("display.width", 180)
        pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

        def p(rel_path: str | Path) -> Path:
            return PROJECT_ROOT / rel_path

        def require_file(rel_path: str | Path) -> Path:
            path = p(rel_path)
            if not path.exists():
                raise FileNotFoundError(f"Missing required project artifact: {rel_path}")
            return path

        def read_table(rel_path: str | Path, **kwargs) -> pd.DataFrame:
            path = require_file(rel_path)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path, **kwargs)
            if suffix in {".pkl", ".pickle"}:
                return pd.read_pickle(path)
            raise ValueError(f"Unsupported table type: {path}")

        def show_image(rel_path: str | Path, width: int = 950) -> None:
            path = require_file(rel_path)
            display(Image(filename=str(path), width=width))

        def show_json(rel_path: str | Path) -> dict:
            path = require_file(rel_path)
            data = json.loads(path.read_text(encoding="utf-8"))
            display(Markdown(f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)[:6000]}\n```"))
            return data

        def show_text_file(rel_path: str | Path, max_chars: int = 7000, language: str = "text") -> str:
            path = require_file(rel_path)
            text = path.read_text(encoding="utf-8")
            clipped = text[:max_chars]
            suffix = "\n\n# ... clipped ..." if len(text) > max_chars else ""
            display(Markdown(f"```{language}\n{clipped}{suffix}\n```"))
            return text

        def show_source_excerpt(rel_path: str | Path, start_text: str, max_lines: int = 90, language: str = "python") -> str:
            path = require_file(rel_path)
            lines = path.read_text(encoding="utf-8").splitlines()
            start_idx = next((i for i, line in enumerate(lines) if start_text in line), None)
            if start_idx is None:
                raise ValueError(f"Could not find marker {start_text!r} in {rel_path}")
            excerpt = "\n".join(lines[start_idx:start_idx + max_lines])
            display(Markdown(f"```{language}\n{excerpt}\n```"))
            return excerpt

        def show_collapsible_source(rel_path: str | Path, title: str, max_chars: int = 20000) -> None:
            if not SHOW_LONG_CODE:
                display(Markdown(f"`SHOW_LONG_CODE=False`，跳过长源码：`{rel_path}`"))
                return
            text = require_file(rel_path).read_text(encoding="utf-8")[:max_chars]
            html = (
                f"<details><summary><strong>{title}</strong> - {rel_path}</summary>"
                f"<pre><code>{text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</code></pre>"
                "</details>"
            )
            display(HTML(html))

        def safe_run_command(command: list[str], cwd: Path | None = None) -> None:
            if not RUN_FULL_PIPELINE:
                print("RUN_FULL_PIPELINE=False，跳过命令：", " ".join(command))
                return
            subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), check=True)

        print("PROJECT_ROOT =", PROJECT_ROOT)
        """
    ))

    cells.append(md(
        r"""
        ## 1. Data and Panel Construction / 数据与面板构建

        这一部分回答三个问题：

        1. 原始数据有哪些，它们在资产定价任务里分别扮演什么角色？
        2. 如何把月末 `t` 的特征、股票池和交易状态对齐到 `t+1` 月收益？
        3. 最终训练面板长什么样？

        主目标变量是：

        ```text
        target_excess_return = target_return(t+1) - rf_next_month
        ```

        主样本使用 `features500/` 和 CSI 500 月度股票池，避免一开始就把工程规模推得太大。
        """
    ))

    cells.append(code(
        r"""
        data_roles = pd.DataFrame([
            ["monthly_returns.pkl", "月度收益", "构造 t+1 目标收益"],
            ["risk_free.csv", "日频无风险利率", "复利聚合为下月 rf_next_month"],
            ["mcap.pkl", "月度市值", "value-weight 组合与规模信息"],
            ["csi500_mask_monthly.pkl", "CSI 500 成分", "定义主样本股票池"],
            ["BLACKLIST.pkl / UNTRADABLE.pkl", "交易限制", "过滤不可交易样本"],
            ["features500/", "公司特征库", "218 个 firm characteristics"],
            ["ind_code.pkl", "静态一级行业分类", "新增 industry edge extension"],
        ], columns=["文件", "内容", "项目用途"])
        display(data_roles)

        panel = read_table("outputs/panels/main_features500_panel.pkl")
        panel["date"] = pd.to_datetime(panel["date"])
        panel_summary = pd.DataFrame({
            "指标": [
                "样本行数", "月份数", "日期范围", "唯一股票数",
                "目标变量", "数值列数量", "每月股票数中位数"
            ],
            "数值": [
                f"{len(panel):,}",
                panel["date"].nunique(),
                f"{panel['date'].min().date()} 至 {panel['date'].max().date()}",
                panel["stock_id"].nunique(),
                "target_excess_return",
                len(panel.select_dtypes(include="number").columns),
                f"{panel.groupby('date')['stock_id'].nunique().median():.0f}",
            ],
        })
        display(panel_summary)
        display(panel.head())
        """
    ))

    cells.append(code(
        r"""
        # Stage 2 metadata：特征数量、缺失处理、保留特征等
        panel_metadata = show_json("outputs/metadata/main_features500_panel_metadata.json")
        kept_features = panel_metadata.get("features", {}).get("kept_features", [])
        print("kept feature count =", len(kept_features))
        print("first 20 features =", kept_features[:20])
        """
    ))

    cells.append(code(
        r"""
        # 数据清洗配置与核心代码摘录
        show_text_file("configs/cleaning_features500.yaml", language="yaml")
        show_source_excerpt("src/data/preprocessing.py", "def build_monthly_panel", max_lines=120)
        show_collapsible_source("src/data/preprocessing.py", "完整 preprocessing.py 源码")
        """
    ))

    cells.append(code(
        r"""
        show_image("reports/figures/figure_1_sample_coverage.png", width=1000)
        """
    ))

    cells.append(md(
        r"""
        ## 2. Benchmark Models / 无图 Benchmark：MLP、IPCA-style、CAE-style

        Benchmark 的设计不是为了凑模型数量，而是为了拆分性能来源：

        - `mlp_predictor`：直接收益预测器，灵活但缺少资产定价解释。
        - `ipca_style`：线性 characteristic-driven beta。
        - `conditional_autoencoder_style`：非线性 characteristic-driven beta。

        这样一来，后续图模型如果有提升，我们可以问：提升来自非线性，还是来自股票之间的关系结构？
        """
    ))

    cells.append(code(
        r"""
        show_text_file("configs/benchmarks_features500.yaml", language="yaml")
        show_source_excerpt("src/models/non_graph_benchmarks.py", "class MLPBenchmark", max_lines=90)
        show_source_excerpt("src/models/non_graph_benchmarks.py", "class IPCAStyleBenchmark", max_lines=120)
        show_source_excerpt("src/models/non_graph_benchmarks.py", "class CAEStyleBenchmark", max_lines=120)
        """
    ))

    cells.append(code(
        r"""
        stage3_metrics = read_table("outputs/metrics/stage3_non_graph_metrics.csv")
        display(stage3_metrics)

        stage3_pred = read_table("outputs/predictions/stage3_non_graph_predictions.pkl")
        stage3_pred["date"] = pd.to_datetime(stage3_pred["date"])
        display(stage3_pred.groupby("model").agg(n_obs=("stock_id", "size"), n_months=("date", "nunique"), date_min=("date", "min"), date_max=("date", "max")).reset_index())
        """
    ))

    cells.append(md(
        r"""
        ## 3. Graph Construction / 图构建：动态相似性图与行业边扩展

        原主规格图是动态混合相似性图，每个月一张图：

        1. `return_correlation`：过去 12 个月收益正相关 kNN。
        2. `feature_cosine_knn`：当月特征余弦相似度 kNN。
        3. `feature_euclidean_knn`：当月标准化特征欧氏距离 kNN。

        后来老师提供 `ind_code.pkl` 后，项目新增了静态一级行业边：

        ```text
        if stock_i and stock_j have same first_industry_code:
            add industry edge with weight = 1.0
        ```

        行业代码只作为边，不作为特征，避免把“行业特征贡献”和“行业关系结构贡献”混在一起。
        """
    ))

    cells.append(code(
        r"""
        show_text_file("configs/graphs_features500.yaml", language="yaml")
        show_source_excerpt("src/graphs/monthly_graphs.py", "def return_correlation_edges", max_lines=75)
        show_source_excerpt("src/graphs/monthly_graphs.py", "def feature_knn_edges", max_lines=75)
        show_source_excerpt("src/graphs/monthly_graphs.py", "def industry_edges", max_lines=70)
        """
    ))

    cells.append(code(
        r"""
        original_graph_stats = read_table("outputs/graphs/features500_similarity_hybrid_stats.csv")
        industry_only_stats = read_table("outputs/industry_extension/graphs/graph_industry_only_stats.csv")
        industry_hybrid_stats = read_table("outputs/industry_extension/graphs/graph_industry_hybrid_stats.csv")

        def graph_stat_summary(stats: pd.DataFrame, label: str) -> pd.DataFrame:
            return (
                stats.groupby("edge_layer")
                .agg(months=("date", "nunique"), mean_edges=("n_edges", "mean"), median_edges=("n_edges", "median"), max_edges=("n_edges", "max"))
                .round(2)
                .assign(graph=label)
                .reset_index()
            )

        graph_stats_all = pd.concat([
            graph_stat_summary(original_graph_stats, "original_similarity_hybrid"),
            graph_stat_summary(industry_only_stats, "industry_only"),
            graph_stat_summary(industry_hybrid_stats, "industry_plus_hybrid"),
        ], ignore_index=True)
        display(graph_stats_all[["graph", "edge_layer", "months", "mean_edges", "median_edges", "max_edges"]])
        """
    ))

    cells.append(code(
        r"""
        show_image("reports/figures/figure_2_graph_overview.png", width=1000)
        show_image("reports/figures/figure_9_industry_graph_extension.png", width=1000)
        """
    ))

    cells.append(md(
        r"""
        ## 4. Graph Latent Factor Model / 图结构增强的条件潜在因子模型

        核心模型不是直接用 GNN 预测收益，而是用图神经网络学习动态暴露：

        ```text
        beta_{i,t} = g_theta(X_t, G_t)
        y_hat_{i,t+1} = beta_{i,t}' mean(f_train)
        ```

        默认模型是两层 GCN beta encoder。训练期学习每个训练月份的 latent factor embedding；样本外预测时使用训练窗口的平均 latent factor premium。
        """
    ))

    cells.append(code(
        r"""
        show_text_file("configs/graph_model_features500.yaml", language="yaml")
        show_source_excerpt("src/models/graph_latent_factor.py", "class GraphBetaEncoder", max_lines=90)
        show_source_excerpt("src/models/graph_latent_factor.py", "class GraphConditionalLatentFactorModel", max_lines=90)
        show_source_excerpt("src/training/graph_model_pipeline.py", "def graph_losses", max_lines=80)
        show_collapsible_source("src/training/graph_model_pipeline.py", "完整 graph_model_pipeline.py 源码")
        """
    ))

    cells.append(md(
        r"""
        ## 5. OOS Evaluation / OOS 预测、排序与定价比较

        评价指标分三类：

        - **OOS R² / RMSE / MAE**：收益预测拟合。
        - **Rank IC / cross-sectional correlation**：横截面排序能力。
        - **Pricing error**：资产定价式诊断。

        重点不是说图模型在所有指标上赢，而是看图结构是否给 conditional beta learning 带来增量信息。
        """
    ))

    cells.append(code(
        r"""
        stage6_summary = read_table("outputs/comparison/stage6_tables/stage6_summary_metrics.csv")
        stage8_main = read_table("outputs/stage8/tables/stage8_main_results_table.csv")
        industry_comp = read_table("outputs/industry_extension/tables/industry_model_comparison.csv")

        main_cols = ["model", "oos_r2_zero_benchmark", "rank_ic_mean", "cross_sectional_corr_mean", "pricing_error_monthly_rmse"]
        display(Markdown("### 主规格 4 模型比较"))
        display(stage8_main[main_cols])

        display(Markdown("### 加入行业扩展后的 6 模型比较"))
        display(industry_comp[main_cols])
        """
    ))

    cells.append(code(
        r"""
        show_image("reports/figures/figure_3_model_comparison.png", width=1000)
        """
    ))

    cells.append(md(
        r"""
        ## 6. Portfolio Backtest / 投资组合回测

        组合构建规则：

        - 使用 month `t` 的预测分数排序。
        - `long-short`：做多顶部十分位，做空底部十分位。
        - `long-only`：只持有顶部十分位。
        - 同时报告 equal-weight 和 value-weight。
        - 主成本设定为 10 bps。

        组合结果是这个项目最重要的经济检验：模型是否真的能转化为可投资信号。
        """
    ))

    cells.append(code(
        r"""
        stage7_perf = read_table("outputs/portfolio/stage7_performance_summary.csv")
        industry_port_compact = read_table("outputs/industry_extension/tables/industry_portfolio_compact.csv")

        display(Markdown("### 主规格组合表现：10 bps"))
        display(stage7_perf.query("transaction_cost_bps == 10")[[
            "model", "strategy_name", "weight_scheme", "annualized_return",
            "annualized_volatility", "sharpe_ratio", "max_drawdown", "avg_monthly_turnover"
        ]].sort_values(["strategy_name", "weight_scheme", "model"]))

        display(Markdown("### 行业扩展组合摘要"))
        display(industry_port_compact)
        """
    ))

    cells.append(code(
        r"""
        show_image("reports/figures/figure_4_portfolio_cumulative.png", width=1000)
        show_image("reports/figures/figure_5_portfolio_summary.png", width=1000)
        show_image("reports/figures/figure_10_industry_portfolio_extension.png", width=1000)
        """
    ))

    cells.append(md(
        r"""
        ## 7. Interpretability and Robustness / 可解释性与稳健性分析

        这部分回答“为什么有效”和“是否只依赖某个单一设定”。

        - Feature-to-exposure association：哪些特征和 latent beta 关系最强？
        - Permutation importance：打乱哪些变量会让 Rank IC 掉得最多？
        - Graph neighborhood diagnostics：top-decile 股票所在邻域是否更稠密？
        - Robustness：换图、换 lookback、换 latent K、换 GAT 后结论是否稳定？
        """
    ))

    cells.append(code(
        r"""
        top_links = read_table("outputs/stage8/tables/stage8_feature_exposure_top_links.csv")
        permutation = read_table("outputs/stage8/tables/stage8_permutation_importance.csv")
        neighbor_mix = read_table("outputs/stage8/tables/stage8_neighbor_edge_mix.csv")
        robustness_pred = read_table("outputs/stage8/tables/stage8_graph_robustness_summary.csv")
        robustness_port = read_table("outputs/stage8/tables/stage8_graph_robustness_portfolio.csv")

        display(Markdown("### Top feature-to-exposure links"))
        display(top_links.head(20))

        display(Markdown("### Graph permutation importance"))
        display(permutation.sort_values("rank_ic_drop", ascending=False).head(15))

        display(Markdown("### Neighbor edge mix"))
        display(neighbor_mix.head(20))

        display(Markdown("### Graph robustness: prediction"))
        display(robustness_pred)

        display(Markdown("### Graph robustness: portfolio"))
        display(robustness_port)
        """
    ))

    cells.append(code(
        r"""
        show_image("reports/figures/figure_6_interpretability.png", width=1100)
        show_image("reports/figures/figure_7_graph_robustness.png", width=1050)
        show_image("reports/figures/figure_8_gat_attention_exploratory.png", width=1000)
        """
    ))

    cells.append(md(
        r"""
        ## 8. Industry Extension / industry extension：补上 proposal 中的静态行业关系

        老师额外提供的 `ind_code.pkl` 让项目可以补上 proposal 里原本设想的“静态行业关系”。

        但这个文件没有日期维度，所以这里不把它说成 point-in-time 行业历史，而是更谨慎地称为 **static industry prior**。

        实证结果的合理解读：

        - `graph_industry_only` 在 OOS R² / pricing RMSE 上比原图略好，说明行业关系确实包含定价/拟合信息。
        - 但原始 dynamic similarity graph 在 Rank IC、截面相关和组合表现上仍明显更强。
        - 因此，行业边不是动态相似性图的替代品；它更像一个稳定但粗粒度的关系先验。
        """
    ))

    cells.append(code(
        r"""
        show_text_file("configs/industry_extension_features500.yaml", language="yaml")
        show_text_file("reports/industry_extension_results.md", language="markdown")
        show_source_excerpt("src/evaluation/industry_extension.py", "def audit_industry_labels", max_lines=90)
        show_source_excerpt("src/evaluation/industry_extension.py", "def run_industry_extension", max_lines=130)
        """
    ))

    cells.append(md(
        r"""
        ## 9. Final Conclusion / 最终结论

        **主结论：**  
        图结构没有让模型在所有统计指标上全面胜出，但它确实为 asset-pricing-oriented conditional beta learning 提供了增量信息。最强证据体现在：

        1. 相比 CAE-style，原图模型显著提升 Rank IC 和 cross-sectional correlation。
        2. 原图模型在 long-only value-weight 组合中取得最高 Sharpe 和收益。
        3. 稳健性结果说明“上图”本身不自动有效，当前表现来自特定的动态混合相似性结构和相对稳定的 GCN beta encoder。

        **行业扩展结论：**  
        静态行业边提供了一部分定价/拟合信息，但没有替代动态相似性图的排序与组合优势。这反而强化了项目的主线：股票关系结构有价值，但关系的定义很重要。

        **最应该对老师强调的表达：**

        > 本项目不是把 GNN 当作黑盒选股器，而是把图结构放进 conditional beta function 中，检验 graph context 是否能改善动态暴露估计，并进一步转化为样本外排序、定价和组合价值。
        """
    ))

    cells.append(md(
        r"""
        ## 10. Full Reproduction Appendix / 完整复现命令

        下面这些 cell 默认不会运行，因为 `RUN_FULL_PIPELINE = False`。  
        如果你需要从头复现，把第一节控制变量改成：

        ```python
        RUN_FULL_PIPELINE = True
        ```

        注意：完整训练会重新生成对应 output 文件，耗时明显更长。
        """
    ))

    cells.append(code(
        r"""
        PY = r"d:\MG\anaconda3\python.exe"
        pipeline_commands = [
            [PY, "scripts/check_environment.py"],
            [PY, "scripts/load_data.py"],
            [PY, "scripts/inspect_data.py"],
            [PY, "scripts/build_panel.py", "--config", "configs/cleaning_features500.yaml"],
            [PY, "scripts/train_benchmarks.py", "--config", "configs/benchmarks_features500.yaml"],
            [PY, "scripts/build_graph.py", "--config", "configs/graphs_features500.yaml"],
            [PY, "scripts/train_graph_model.py", "--config", "configs/graph_model_features500.yaml"],
            [PY, "scripts/evaluate_model_comparison.py", "--config", "configs/evaluation_features500.yaml"],
            [PY, "scripts/backtest_portfolio.py", "--config", "configs/portfolio_features500.yaml"],
            [PY, "scripts/run_stage8_analysis.py", "--config", "configs/stage8_features500.yaml"],
            [PY, "scripts/generate_report_figures.py", "--config", "configs/report_figures.yaml"],
            [PY, "scripts/run_industry_extension.py", "--config", "configs/industry_extension_features500.yaml"],
        ]

        for cmd in pipeline_commands:
            safe_run_command(cmd)
        """
    ))

    cells.append(md(
        r"""
        ## 11. 附录：关键文件地图

        - `Proposal.md`：研究问题和最初设计。
        - `src/data/preprocessing.py`：数据清洗与面板构建。
        - `src/models/non_graph_benchmarks.py`：MLP、IPCA-style、CAE-style。
        - `src/graphs/monthly_graphs.py`：动态相似性图与 industry edge。
        - `src/models/graph_latent_factor.py`：GCN/GAT beta encoder。
        - `src/training/graph_model_pipeline.py`：图模型训练、loss、OOS 预测。
        - `src/portfolio/backtest.py`：long-short / long-only 回测。
        - `src/evaluation/stage8_analysis.py`：解释性与稳健性。
        - `src/evaluation/industry_extension.py`：新增行业分类扩展。
        """
    ))

    cells.append(code(
        r"""
        show_collapsible_source("src/models/non_graph_benchmarks.py", "完整 non_graph_benchmarks.py 源码")
        show_collapsible_source("src/graphs/monthly_graphs.py", "完整 monthly_graphs.py 源码")
        show_collapsible_source("src/models/graph_latent_factor.py", "完整 graph_latent_factor.py 源码")
        show_collapsible_source("src/portfolio/backtest.py", "完整 portfolio/backtest.py 源码")
        show_collapsible_source("src/evaluation/industry_extension.py", "完整 industry_extension.py 源码")
        """
    ))

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Saved notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
