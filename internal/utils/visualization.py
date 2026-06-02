import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


PLOTS_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"


def _save(fig, filename: str):
    if filename:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_regression_fit(
    x_pts, y_pts, x_grid, y_true, preds: dict, title,
    filename=None, xlabel='x', ylabel='y', figsize=(8, 5)
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_pts, y_pts, s=14, color='gray', alpha=0.55,
               label='Зашумлённые данные', zorder=1)
    ax.plot(x_grid, y_true, '--', color='black', linewidth=2.0,
            label='Истинная зависимость', zorder=3)
    cmap = plt.get_cmap('tab10')
    for i, (label, y_pred) in enumerate(preds.items()):
        ax.plot(x_grid, y_pred, '-', color=cmap(i % 10), linewidth=1.8,
                label=label, zorder=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, filename)


def plot_loss_history(
    series: dict, title, xlabel='Эпоха',
    ylabel='Значение функции потерь', filename=None,
    logy=True, figsize=(8, 5)
):
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap('tab10')
    for i, (label, xy) in enumerate(series.items()):
        xs, ys = xy
        ax.plot(xs, ys, '-', color=cmap(i % 10), linewidth=1.7,
                markersize=3, label=label)
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=8, framealpha=0.85)
    plt.tight_layout()
    _save(fig, filename)


def plot_coeff_bars(
    groups: dict, title, filename=None,
    xlabel='Индекс коэффициента', ylabel='|вес|',
    logy=False, figsize=(8, 5)
):
    fig, ax = plt.subplots(figsize=figsize)
    names = list(groups.keys())
    p = max(len(v) for v in groups.values())
    idx = np.arange(p)
    width = 0.8 / max(len(names), 1)
    cmap = plt.get_cmap('tab10')
    for i, name in enumerate(names):
        vals = np.asarray(groups[name], dtype=float)
        padded = np.zeros(p)
        padded[:len(vals)] = np.abs(vals)
        ax.bar(idx + i * width, padded, width, label=name, color=cmap(i % 10))
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_xticks(idx + width * (len(names) - 1) / 2)
    ax.set_xticklabels([str(j) for j in idx])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, axis='y', alpha=0.35)
    plt.tight_layout()
    _save(fig, filename)


def plot_metric_vs_param(
    xvals, series: dict, title,
    xlabel, ylabel='Число итераций',
    filename=None, logx=False, logy=False
):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(xvals, ys, 'o-', label=name, color=cmap(i % 10),
                linewidth=1.8, markersize=4)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which='both', alpha=0.4)
    ax.legend(fontsize=7, ncol=2, framealpha=0.85)
    plt.tight_layout()
    _save(fig, filename)


def plot_iter_vs_step(steps, iters, title, xlabel='Шаг α', filename=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, iters, 'o-', color='steelblue', linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel('Число итераций (log)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, filename)


def print_table(headers, rows, title="", slug=None):
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    col_w = [max(len(str(h)),
                 max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("─" * w for w in col_w))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print()

    import csv
    tables_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if slug:
        safe = slug
    else:
        safe = "".join(c if c.isalnum() or c in " _" else "_"
                       for c in (title or "table")).strip().replace(" ", "_")
    with open(tables_dir / f"{safe}.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(headers)
        for r in rows:
            w.writerow([str(c) for c in r])

    tex_path = tables_dir / f"{safe}.tex"

    def _escape(s: str) -> str:
        repl = {
            "\\": r"\textbackslash{}",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "&": r"\&",
            "κ": r"$\kappa$",
            "α": r"$\alpha$",
            "ε": r"$\varepsilon$",
            "λ": r"$\lambda$",
            "σ": r"$\sigma$",
            "✓": r"\checkmark",
            "✗": r"$\times$",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
        return s

    with open(tex_path, "w", encoding="utf-8") as fp:
        fp.write("\\begin{table}[H]\n")
        fp.write("\\centering\n")
        if title:
            fp.write(f"\\caption{{{_escape(title)}}}\n")
            fp.write(f"\\label{{tab:{safe}}}\n")
        colspec = 'l' + ''.join('c' for _ in range(len(headers)-1))
        fp.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        fp.write("\\toprule\n")
        fp.write(' & '.join(_escape(str(h)) for h in headers) + " \\\\ \n")
        fp.write("\\midrule\n")
        for r in rows:
            fp.write(' & '.join(_escape(str(c)) for c in r) + " \\\\ \n")
        fp.write("\\bottomrule\n")
        fp.write("\\end{tabular}\n")
        fp.write("\\end{table}\n")
