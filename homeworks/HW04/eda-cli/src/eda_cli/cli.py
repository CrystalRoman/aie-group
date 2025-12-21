from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_histograms_per_column,
    plot_missing_matrix,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(path: Path, sep: str, encoding: str) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    return pd.read_csv(path, sep=sep, encoding=encoding)


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу"),
    sep: str = typer.Option(",", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", help="Кодировка"),
) -> None:
    df = _load_csv(Path(path), sep, encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу"),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта"),
    sep: str = typer.Option(",", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", help="Кодировка"),
    max_hist_columns: int = typer.Option(6, help="Макс. число гистограмм"),
    top_k_categories: int = typer.Option(5, help="Top-K категорий"),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта"),
    json_summary: bool = typer.Option(False, help="Сохранить summary.json"),
    min_quality_score: float = typer.Option(
        0.5, help="Минимально допустимое качество данных"
    ),
    fail_on_low_quality: bool = typer.Option(
        False, help="Завершить с ошибкой, если качество ниже порога"
    ),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep, encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    quality_flags = compute_quality_flags(summary, missing_df)

    if fail_on_low_quality and quality_flags["quality_score"] < min_quality_score:
        raise typer.Exit(
            code=1,
        )

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv")
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv")
    save_top_categories_tables(top_cats, out_root / "top_categories")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")
        f.write("## Качество данных\n\n")
        for k, v in quality_flags.items():
            f.write(f"- {k}: **{v}**\n")

    if json_summary:
        problems = [
            c.name
            for c in summary.columns
            if c.missing_share > 0.5 or c.unique <= 1
        ]
        with (out_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_rows": summary.n_rows,
                    "n_cols": summary.n_cols,
                    "quality_score": quality_flags["quality_score"],
                    "problem_columns": problems,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сохранён в {out_root}")


if __name__ == "__main__":
    app()
