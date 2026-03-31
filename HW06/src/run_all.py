from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class Config:
    data_dir: str = "course/testfiles/data"
    out_dir: str = "HW06/out"
    rf: float = 0.04
    seed: int = 100000


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, float_format="%.15f")


def must_exist(data_dir: Path, name: str) -> Path:
    p = data_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


# Task 10 helpers

def read_cov_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    cols = ["x1", "x2", "x3", "x4", "x5"]

    if not all(c in df.columns for c in cols):
        raise ValueError(f"Expected columns {cols} in {path.name}, got {list(df.columns)}")

    cov = df[cols].to_numpy(dtype=float)

    if cov.shape != (5, 5):
        raise ValueError(f"Expected 5x5 covariance matrix in {path.name}, got {cov.shape}")

    return cov, cols


def read_means(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    if "Mean" not in df.columns:
        raise ValueError(f"Expected column 'Mean' in {path.name}, got {list(df.columns)}")

    mu = df["Mean"].to_numpy(dtype=float)

    if len(mu) != 5:
        raise ValueError(f"Expected 5 means in {path.name}, got {len(mu)}")

    return mu


def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def component_std(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma_p = portfolio_vol(w, cov)
    if sigma_p <= 0:
        return np.full_like(w, 1e10, dtype=float)
    return w * (cov @ w) / sigma_p


def sse_equal_risk(w: np.ndarray, cov: np.ndarray) -> float:
    csd = component_std(w, cov)
    target = np.mean(csd)
    return float(np.sum((csd - target) ** 2))


def sse_budgeted_risk(w: np.ndarray, cov: np.ndarray, budget: np.ndarray) -> float:
    csd = component_std(w, cov)
    adj = csd / budget
    target = np.mean(adj)
    return float(np.sum((adj - target) ** 2))


def neg_sharpe_ratio(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    port_ret = float(w @ mu)
    port_vol = portfolio_vol(w, cov)
    if port_vol <= 0:
        return 1e10
    return -((port_ret - rf) / port_vol)


def project_to_box_simplex(w: np.ndarray, lower: float, upper: float, iters: int = 200) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w = np.clip(w, lower, upper)

    for _ in range(iters):
        diff = 1.0 - w.sum()
        if abs(diff) < 1e-12:
            break

        if diff > 0:
            free = np.where(w < upper - 1e-12)[0]
            if len(free) == 0:
                break
            w[free] += diff / len(free)
            w = np.minimum(w, upper)
        else:
            free = np.where(w > lower + 1e-12)[0]
            if len(free) == 0:
                break
            w[free] -= (-diff) / len(free)
            w = np.maximum(w, lower)

    return w


def save_weights(w: np.ndarray, out_file: Path) -> pd.DataFrame:
    df = pd.DataFrame({"W": w})
    write_csv(out_file, df)
    return df


def solve_risk_parity(cov: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    n = cov.shape[0]
    w0 = inverse_vol_weights(cov)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    objective = (
        (lambda w: sse_equal_risk(w, cov))
        if budget is None
        else (lambda w: sse_budgeted_risk(w, cov, budget))
    )

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-15},
    )

    if not res.success:
        raise RuntimeError(f"Risk parity optimization failed: {res.message}")

    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def feasible_starts(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
    n_random: int = 30,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []

    w_eq = np.full(len(mu), 1.0 / len(mu))
    if np.all(w_eq >= lower - 1e-12) and np.all(w_eq <= upper + 1e-12):
        starts.append(w_eq)

    starts.append(project_to_box_simplex(inverse_vol_weights(cov), lower, upper))

    excess = mu - rf
    pos = np.maximum(excess, 0.0)
    if pos.sum() > 0:
        starts.append(project_to_box_simplex(pos / pos.sum(), lower, upper))

    for _ in range(n_random):
        z = rng.random(len(mu))
        z = z / z.sum()
        starts.append(project_to_box_simplex(z, lower, upper))

    return starts


def solve_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(lower, upper)] * len(mu)

    best_w = None
    best_obj = np.inf

    for w0 in feasible_starts(mu, cov, rf, lower, upper, rng):
        res = minimize(
            lambda w: neg_sharpe_ratio(w, mu, cov, rf),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-15},
        )

        w = project_to_box_simplex(np.asarray(res.x, dtype=float), lower, upper)

        if (
            abs(w.sum() - 1.0) <= 1e-6
            and np.all(w >= lower - 1e-6)
            and np.all(w <= upper + 1e-6)
        ):
            obj = neg_sharpe_ratio(w, mu, cov, rf)
            if obj < best_obj:
                best_obj = obj
                best_w = w

    if best_w is None:
        raise RuntimeError("Max Sharpe optimization failed.")

    return best_w


# Task 11 helpers

def total_return(x: np.ndarray) -> float:
    return float(np.prod(1.0 + x) - 1.0)


def carino_k(rp: np.ndarray) -> np.ndarray:
    portfolio_total = total_return(rp)

    if abs(portfolio_total) < 1e-15:
        return np.ones_like(rp, dtype=float)

    K = np.log1p(portfolio_total) / portfolio_total
    return np.array([
        1.0 / K if abs(rpt) < 1e-15 else np.log1p(rpt) / (K * rpt)
        for rpt in rp
    ], dtype=float)


def update_weights_through_time(w0: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, n = returns.shape
    weights_t = np.zeros((T, n), dtype=float)
    portfolio_r = np.zeros(T, dtype=float)

    w_curr = w0.copy()

    for t in range(T):
        weights_t[t] = w_curr
        portfolio_r[t] = float(np.dot(w_curr, returns[t]))

        w_grow = w_curr * (1.0 + returns[t])
        w_curr = w_grow / (1.0 + portfolio_r[t])

    return weights_t, portfolio_r


def vol_attr_from_contrib(contrib: np.ndarray, rp: np.ndarray) -> np.ndarray:
    sigma_p = float(np.std(rp, ddof=1))
    if sigma_p <= 0:
        return np.zeros(contrib.shape[1], dtype=float)

    return np.array([
        float(np.cov(contrib[:, j], rp, ddof=1)[0, 1] / sigma_p)
        for j in range(contrib.shape[1])
    ])


# Task 10

def task_10_1(cfg: Config, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    cov, _ = read_cov_matrix(must_exist(data_dir, "test5_2.csv"))
    w = solve_risk_parity(cov)
    return save_weights(w, out_dir / "testout10_1.csv")


def task_10_2(cfg: Config, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    cov, _ = read_cov_matrix(must_exist(data_dir, "test5_2.csv"))
    budget = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=float)
    w = solve_risk_parity(cov, budget=budget)
    return save_weights(w, out_dir / "testout10_2.csv")


def task_10_3(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    cov, _ = read_cov_matrix(must_exist(data_dir, "test5_2.csv"))
    mu = read_means(must_exist(data_dir, "test10_3_means.csv"))
    w = solve_max_sharpe(mu, cov, rf=cfg.rf, lower=1e-6, upper=1.0, rng=rng)
    return save_weights(w, out_dir / "testout10_3.csv")


def task_10_4(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    cov, _ = read_cov_matrix(must_exist(data_dir, "test5_2.csv"))
    mu = read_means(must_exist(data_dir, "test10_3_means.csv"))
    w = solve_max_sharpe(mu, cov, rf=cfg.rf, lower=0.1, upper=0.5, rng=rng)
    return save_weights(w, out_dir / "testout10_4.csv")


# Task 11

def task_11_1(cfg: Config, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    returns = pd.read_csv(must_exist(data_dir, "test11_1_returns.csv"))
    weights = pd.read_csv(must_exist(data_dir, "test11_1_weights.csv"))

    asset_cols = list(returns.columns)
    r = returns[asset_cols].to_numpy(dtype=float)

    if len(weights.columns) != 1:
        raise ValueError("Expected test11_1_weights.csv to have exactly one weight column.")

    w0 = weights.iloc[:, 0].to_numpy(dtype=float)
    if len(w0) != len(asset_cols):
        raise ValueError(f"Weight length {len(w0)} does not match number of assets {len(asset_cols)}.")

    w_t, rp = update_weights_through_time(w0, r)
    contrib = w_t * r

    k = carino_k(rp)
    portfolio_total = total_return(rp)
    sigma_p = float(np.std(rp, ddof=1))

    total_row = pd.Series(
        {asset_cols[i]: total_return(r[:, i]) for i in range(len(asset_cols))},
        name="TotalReturn",
    )

    ret_attr = pd.Series(
        (k[:, None] * contrib).sum(axis=0),
        index=asset_cols,
        name="Return Attribution",
    )

    vol_attr = pd.Series(
        vol_attr_from_contrib(contrib, rp),
        index=asset_cols,
        name="Vol Attribution",
    )

    out = pd.concat(
        [total_row.to_frame().T, ret_attr.to_frame().T, vol_attr.to_frame().T],
        axis=0,
    )
    out.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    out["Portfolio"] = [portfolio_total, portfolio_total, sigma_p]

    out = out.reset_index(drop=True)
    write_csv(out_dir / "testout11_1.csv", out)
    return out


def task_11_2(cfg: Config, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    factor_r = pd.read_csv(must_exist(data_dir, "test11_2_factor_returns.csv"))
    stock_r = pd.read_csv(must_exist(data_dir, "test11_2_stock_returns.csv"))
    beta = pd.read_csv(must_exist(data_dir, "test11_2_beta.csv"))
    weights = pd.read_csv(must_exist(data_dir, "test11_2_weights.csv"))

    factor_names = list(factor_r.columns)
    stock_names = list(stock_r.columns)

    F = factor_r[factor_names].to_numpy(dtype=float)   # T x m
    R = stock_r[stock_names].to_numpy(dtype=float)     # T x n

    if "Stock" not in beta.columns:
        raise ValueError("Expected a 'Stock' column in test11_2_beta.csv.")

    B = beta.set_index("Stock").loc[stock_names, factor_names].to_numpy(dtype=float)  # n x m

    if len(weights.columns) != 1:
        raise ValueError("Expected test_11_2_weights.csv to have exactly one weight column.")

    w0 = weights.iloc[:, 0].to_numpy(dtype=float)
    if len(w0) != len(stock_names):
        raise ValueError(f"Weight length {len(w0)} does not match number of stocks {len(stock_names)}.")

    stock_w_t, rp = update_weights_through_time(w0, R)

    factor_w_t = stock_w_t @ B
    factor_contrib = factor_w_t * F

    fitted = F @ B.T
    resid = R - fitted
    alpha_t = np.sum(stock_w_t * resid, axis=1)

    k = carino_k(rp)
    portfolio_total = total_return(rp)
    sigma_p = float(np.std(rp, ddof=1))

    total_vals = {factor_names[j]: total_return(F[:, j]) for j in range(len(factor_names))}
    total_vals["Alpha"] = total_return(alpha_t)
    total_row = pd.Series(total_vals, name="TotalReturn")

    ret_vals = {factor_names[j]: float(np.sum(k * factor_contrib[:, j])) for j in range(len(factor_names))}
    ret_vals["Alpha"] = float(np.sum(k * alpha_t))
    ret_attr = pd.Series(ret_vals, name="Return Attribution")

    vol_vals = {factor_names[j]: float(np.cov(factor_contrib[:, j], rp, ddof=1)[0, 1] / sigma_p) for j in range(len(factor_names))}
    vol_vals["Alpha"] = float(np.cov(alpha_t, rp, ddof=1)[0, 1] / sigma_p)
    vol_attr = pd.Series(vol_vals, name="Vol Attribution")

    out = pd.concat(
        [total_row.to_frame().T, ret_attr.to_frame().T, vol_attr.to_frame().T],
        axis=0,
    )
    out.insert(0, "Value", ["TotalReturn", "Return Attribution", "Vol Attribution"])
    out["Portfolio"] = [portfolio_total, portfolio_total, sigma_p]

    out = out.reset_index(drop=True)
    write_csv(out_dir / "testout11_2.csv", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rf", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir or Config.data_dir,
        out_dir=args.out_dir or Config.out_dir,
        rf=args.rf if args.rf is not None else Config.rf,
        seed=args.seed if args.seed is not None else Config.seed,
    )

    data_dir = Path(cfg.data_dir)
    out_dir = ensure_dir(cfg.out_dir)
    rng = np.random.default_rng(cfg.seed)

    results: Dict[str, Any] = {}
    results["10.1"] = task_10_1(cfg, data_dir, out_dir).to_dict(orient="records")
    results["10.2"] = task_10_2(cfg, data_dir, out_dir).to_dict(orient="records")
    results["10.3"] = task_10_3(cfg, data_dir, out_dir, rng).to_dict(orient="records")
    results["10.4"] = task_10_4(cfg, data_dir, out_dir, rng).to_dict(orient="records")
    results["11.1"] = task_11_1(cfg, data_dir, out_dir).to_dict(orient="records")
    results["11.2"] = task_11_2(cfg, data_dir, out_dir).to_dict(orient="records")

    if not args.quiet:
        print("Done. Outputs written to:", out_dir.resolve())
        for fn in [
            "testout10_1.csv",
            "testout10_2.csv",
            "testout10_3.csv",
            "testout10_4.csv",
            "testout11_1.csv",
            "testout11_2.csv",
        ]:
            p = out_dir / fn
            print(" -", fn, "OK" if p.exists() else "MISSING")


if __name__ == "__main__":
    main()