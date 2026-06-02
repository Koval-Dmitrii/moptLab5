import numpy as np

from internal.data import make_near_linear, make_nonlinear
from internal.models import PolynomialModel
from internal.losses import RegressionProblem
from internal.optimizers import (
    analytic_linear_1d,
    sgd,
    minibatch_gd,
    gauss_newton,
    levenberg_marquardt,
)
from internal.utils import (
    plot_regression_fit,
    plot_loss_history,
    plot_coeff_bars,
    plot_metric_vs_param,
    print_table,
)


DEGREES = [1, 2, 3, 4, 5]
METHODS = ["Аналит", "SGD", "MB", "GN", "LM"]


def _problem(dataset, degree, l1=0.0, l2=0.0):
    model = PolynomialModel(degree).fit(dataset.x)
    Phi = model.design(dataset.x)
    prob = RegressionProblem(Phi, dataset.y, l1=l1, l2=l2)
    return model, prob


def _run_methods(prob, n_params, degree, seed=0):
    w0 = np.zeros(n_params)
    out = {}
    out["GN"] = gauss_newton(prob, w0)
    out["LM"] = levenberg_marquardt(prob, w0)
    out["SGD"] = sgd(prob, w0, lr0=0.08, decay=0.01, n_epochs=400, seed=seed)
    out["MB"] = minibatch_gd(prob, w0, batch_size=16, lr0=0.25,
                             decay=0.003, n_epochs=400, seed=seed)
    if degree == 1:
        out["Аналит"] = analytic_linear_1d(prob)
    return out


def baseline_models():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1: Базовые модели без регуляризации")
    print("=" * 70)

    for dataset in [make_near_linear(), make_nonlinear()]:
        per_degree = {}
        risk_rows = []
        for degree in DEGREES:
            model, prob = _problem(dataset, degree)
            res = _run_methods(prob, model.n_params, degree)
            per_degree[degree] = (model, res)
            row = [degree]
            for mkey in METHODS:
                row.append(f"{res[mkey].risk:.4f}" if mkey in res else "—")
            risk_rows.append(row)
        print_table(["Степень"] + METHODS, risk_rows,
                    title=f"{dataset.name}: итоговый эмпирический риск Q",
                    slug=f"base_{dataset.slug}_risk")

        xg, yt = dataset.grid()

        preds_deg = {}
        for degree in [1, 3, 5]:
            model, res = per_degree[degree]
            preds_deg[f"степень {degree}"] = model.predict(xg, res["GN"].w)
        plot_regression_fit(
            dataset.x, dataset.y, xg, yt, preds_deg,
            f"{dataset.name}: полиномы разных степеней (Гаусс-Ньютон)",
            filename=f"fig_{dataset.slug}_degrees.png")

        rep = 1 if dataset.slug == "linear" else 5
        model, res = per_degree[rep]
        preds_m = {mkey: model.predict(xg, r.w) for mkey, r in res.items()}
        plot_regression_fit(
            dataset.x, dataset.y, xg, yt, preds_m,
            f"{dataset.name}: сравнение методов (степень {rep})",
            filename=f"fig_{dataset.slug}_methods.png")

        series = {}
        for mkey in ["SGD", "MB"]:
            h = res[mkey].history
            series[mkey] = (h["epoch"], h["loss"])
        plot_loss_history(
            series, f"{dataset.name}: динамика потерь (степень {rep})",
            xlabel="Эпоха", filename=f"fig_{dataset.slug}_loss.png")


def batch_size_study():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2: Влияние размера batch")
    print("=" * 70)

    dataset = make_nonlinear()
    degree = 5
    model, prob = _problem(dataset, degree)
    w0 = np.zeros(model.n_params)
    m = prob.m
    sizes = [1, 4, 8, 16, 32, 64, m]

    rows = []
    by_epoch = {}
    by_grad = {}
    for bs in sizes:
        r = minibatch_gd(prob, w0, batch_size=bs, lr0=0.05, decay=0.005,
                         n_epochs=200, seed=0)
        label = f"batch={bs}" if bs != m else f"batch=m ({m})"
        by_epoch[label] = (r.history["epoch"], r.history["loss"])
        by_grad[label] = (r.history["grad"], r.history["loss"])
        rows.append([
            bs, f"{r.risk:.4e}", r.n_epoch, r.n_grad,
            f"{r.time:.3f}", "✓" if r.converged else "✗",
        ])
    print_table(
        ["batch", "Итоговый риск", "Эпохи", "grad_calls", "Время, с", "Сошлось"],
        rows, title="Влияние размера batch (нелинейная, степень 5)",
        slug="batch_size")

    plot_loss_history(by_epoch, "Динамика потерь по эпохам (разные batch)",
                      xlabel="Эпоха", filename="fig_batch_epoch.png")
    plot_loss_history(by_grad, "Динамика потерь по числу вычислений градиента",
                      xlabel="Число вычислений градиента (батчей)",
                      filename="fig_batch_grad.png")


def _full_batch(prob, w0, n_epochs=600, lr0=0.12):
    return minibatch_gd(prob, w0, batch_size=prob.m, lr0=lr0, decay=0.001,
                        n_epochs=n_epochs, seed=0, record_every=5)


def regularization_study():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 3: Влияние регуляризации")
    print("=" * 70)

    dataset = make_nonlinear()
    degree = 9
    model, prob_base = _problem(dataset, degree)
    w0 = np.zeros(model.n_params)

    base = gauss_newton(prob_base, w0)

    variants = {
        "L1":          dict(l1=0.05, l2=0.0),
        "L2":          dict(l1=0.0,  l2=0.05),
        "Elastic Net": dict(l1=0.02, l2=0.02),
    }

    fitted = {"База (без рег.)": base.w}
    dyn_files = {"L1": "fig_reg_dyn_l1.png", "L2": "fig_reg_dyn_l2.png",
                 "Elastic Net": "fig_reg_dyn_en.png"}

    summary_rows = []
    for name, p in variants.items():
        _, prob = _problem(dataset, degree, l1=p["l1"], l2=p["l2"])
        r = _full_batch(prob, w0)
        fitted[name] = r.w
        nz = int(np.sum(np.abs(r.w[1:]) > 1e-3))
        summary_rows.append([
            name, f"{p['l1']:.3f}", f"{p['l2']:.3f}",
            f"{r.risk:.4e}", f"{r.reg:.4e}", f"{r.loss:.4e}", nz,
        ])
        h = r.history
        plot_loss_history(
            {"Полные потери L": (h["epoch"], h["loss"]),
             "Эмпирический риск Q": (h["epoch"], h["risk"]),
             "Регуляризатор": (h["epoch"], np.maximum(h["reg"], 1e-12))},
            f"Динамика потерь: {name} (степень {degree})",
            xlabel="Эпоха", filename=dyn_files[name])

    print_table(
        ["Тип", "λ1", "λ2", "Риск Q", "Рег.", "Потери L", "Ненулевых"],
        summary_rows, title="Сводка по регуляризации (нелинейная, степень 9)",
        slug="reg_summary")

    xg, yt = dataset.grid()
    preds = {k: model.predict(xg, w) for k, w in fitted.items()}
    plot_regression_fit(
        dataset.x, dataset.y, xg, yt, preds,
        f"Влияние регуляризации на аппроксимацию (степень {degree})",
        filename="fig_reg_fit.png")

    plot_coeff_bars(
        {k: np.abs(w) for k, w in fitted.items()},
        f"Модули коэффициентов до и после регуляризации (степень {degree})",
        filename="fig_reg_coeffs.png", logy=True)

    coeff_rows = []
    for j in range(model.n_params):
        coeff_rows.append([
            j] + [f"{abs(fitted[k][j]):.3f}" for k in fitted])
    print_table(
        ["Индекс"] + list(fitted.keys()), coeff_rows,
        title="Модули коэффициентов |w_j|", slug="reg_coeffs")


def _raw_problem(dataset, degree, l2=0.0):
    mean = float(np.mean(dataset.x))
    std = float(np.std(dataset.x)) or 1.0
    z = (dataset.x - mean) / std
    Phi = np.vander(z, degree + 1, increasing=True)
    return RegressionProblem(Phi, dataset.y, l2=l2)


def methods_comparison():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 4: Сравнение методов оптимизации")
    print("=" * 70)

    setups = [
        ("Почти линейная, степень 1", make_near_linear(), 1),
        ("Нелинейная, степень 5", make_nonlinear(), 5),
    ]
    rows = []
    for title, dataset, degree in setups:
        model, prob = _problem(dataset, degree)
        res = _run_methods(prob, model.n_params, degree)
        for mkey in METHODS:
            if mkey not in res:
                continue
            r = res[mkey]
            steps = r.n_epoch if mkey in ("SGD", "MB") else r.n_iter
            rows.append([
                title, mkey, f"{r.risk:.4e}", steps, r.n_grad,
                f"{r.time:.3f}", "✓" if r.converged else "✗",
            ])
    print_table(
        ["Постановка", "Метод", "Риск Q", "Итер/эпох", "grad_calls",
         "Время, с", "Сошлось"],
        rows, title="Сравнение методов оптимизации", slug="methods_compare")

    print("\n--- Гаусс-Ньютон против Левенберга-Марквардта (плохая обусловленность) ---")
    dataset = make_nonlinear()
    gl_rows = []
    series = {}
    for degree in [8, 12]:
        prob = _raw_problem(dataset, degree)
        w0 = np.zeros(degree + 1)
        kappa = float(np.linalg.cond(prob.hessian(w0)))
        rg = gauss_newton(prob, w0, max_iter=100)
        rl = levenberg_marquardt(prob, w0, max_iter=200)
        gl_rows.append([
            degree, f"{kappa:.2e}", "GN", f"{rg.risk:.4e}",
            rg.n_iter, rg.n_grad, "✓" if rg.converged else "✗"])
        gl_rows.append([
            degree, f"{kappa:.2e}", "LM", f"{rl.risk:.4e}",
            rl.n_iter, rl.n_grad, "✓" if rl.converged else "✗"])
        series[f"GN, степень {degree}"] = (rg.history["iter"],
                                           np.maximum(rg.history["loss"], 1e-12))
        series[f"LM, степень {degree}"] = (rl.history["iter"],
                                           np.maximum(rl.history["loss"], 1e-12))
    print_table(
        ["Степень", "κ(H)", "Метод", "Риск Q", "Итерации", "grad_calls", "Сошлось"],
        gl_rows, title="Гаусс-Ньютон и Левенберг-Марквардт при плохой обусловленности",
        slug="gn_lm_illcond")

    plot_loss_history(
        series, "Сходимость GN и LM (необусловленный базис Вандермонда)",
        xlabel="Итерация", filename="fig_gn_lm.png")
