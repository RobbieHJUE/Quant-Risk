import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize


# =========================
# Paths
# =========================
HW_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = HW_DIR / "out"
DATA_DIR = HW_DIR.parents[0] / "course" / "testfiles" / "data"

COV_FILE = DATA_DIR / "test5_2.csv"
MEAN_FILE = DATA_DIR / "test10_3_means.csv"

OUT_10_1 = OUT_DIR / "testout10_1.csv"
OUT_10_2 = OUT_DIR / "testout10_2.csv"
OUT_10_3 = OUT_DIR / "testout10_3.csv"
OUT_10_4 = OUT_DIR / "testout10_4.csv"


# =========================
# Read data
# =========================
def read_cov_matrix(path: Path):
    df = pd.read_csv(path)

    # 只保留 x1-x5，并按这个顺序
    cols = ["x1", "x2", "x3", "x4", "x5"]
    df = df[cols].copy()

    cov = df.to_numpy(dtype=float)

    if cov.shape != (5, 5):
        raise ValueError("test5_2.csv must be a 5x5 covariance matrix with columns x1-x5")

    return cov, cols


def read_means(path: Path):
    df = pd.read_csv(path)
    if "Mean" not in df.columns:
        raise ValueError("test10_3_means.csv must contain a column named 'Mean'")

    mu = df["Mean"].to_numpy(dtype=float)
    if len(mu) != 5:
        raise ValueError("test10_3_means.csv must contain exactly 5 rows")

    return mu


# =========================
# Math helpers
# =========================
def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    val = float(w @ cov @ w)
    val = max(val, 0.0)
    return float(np.sqrt(val))


def inverse_vol_start(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def component_std(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    CSD = w * (Sigma w) / sqrt(w' Sigma w)
    """
    sigma_p = portfolio_vol(w, cov)
    if sigma_p <= 0:
        return np.full_like(w, 1e10, dtype=float)
    return w * (cov @ w) / sigma_p


def sse_equal_risk(w: np.ndarray, cov: np.ndarray) -> float:
    csd = component_std(w, cov)
    target = np.mean(csd)
    return float(np.sum((csd - target) ** 2))


def sse_budgeted_risk(w: np.ndarray, cov: np.ndarray, budget: np.ndarray) -> float:
    """
    Adjusted CSD = CSD / budget
    minimize SSE of adjusted CSD
    """
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


# =========================
# Feasible start helper
# =========================
def project_to_box_simplex(w: np.ndarray, lower: float, upper: float, iters: int = 200) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    n = len(w)

    w = np.clip(w, lower, upper)

    for _ in range(iters):
        s = w.sum()
        diff = 1.0 - s

        if abs(diff) < 1e-12:
            break

        if diff > 0:
            free = np.where(w < upper - 1e-12)[0]
            if len(free) == 0:
                break
            add = diff / len(free)
            w[free] += add
            w = np.minimum(w, upper)
        else:
            free = np.where(w > lower + 1e-12)[0]
            if len(free) == 0:
                break
            sub = (-diff) / len(free)
            w[free] -= sub
            w = np.maximum(w, lower)

    return w


# =========================
# Solvers
# =========================
def solve_risk_parity(cov: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    n = cov.shape[0]
    w0 = inverse_vol_start(cov)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    if budget is None:
        objective = lambda w: sse_equal_risk(w, cov)
    else:
        budget = np.asarray(budget, dtype=float)
        objective = lambda w: sse_budgeted_risk(w, cov, budget)

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
    w = w / w.sum()
    return w


def generate_feasible_starts(mu: np.ndarray, cov: np.ndarray, rf: float,
                             lower: float, upper: float, n_random: int = 30):
    n = len(mu)
    starts = []

    # equal weight
    w_eq = np.full(n, 1.0 / n)
    if np.all(w_eq >= lower - 1e-12) and np.all(w_eq <= upper + 1e-12):
        starts.append(w_eq)

    # inverse vol
    w_iv = inverse_vol_start(cov)
    w_iv = project_to_box_simplex(w_iv, lower, upper)
    starts.append(w_iv)

    # positive excess mean
    excess = mu - rf
    pos = np.maximum(excess, 0.0)
    if pos.sum() > 0:
        w_mu = pos / pos.sum()
        w_mu = project_to_box_simplex(w_mu, lower, upper)
        starts.append(w_mu)

    # random starts
    rng = np.random.default_rng(123)
    for _ in range(n_random):
        z = rng.random(n)
        z = z / z.sum()
        z = project_to_box_simplex(z, lower, upper)
        starts.append(z)

    return starts


def solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float,
                     lower: float, upper: float) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(lower, upper)] * len(mu)

    starts = generate_feasible_starts(mu, cov, rf, lower, upper, n_random=30)

    best_w = None
    best_obj = np.inf

    for w0 in starts:
        res = minimize(
            lambda w: neg_sharpe_ratio(w, mu, cov, rf),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 5000, "ftol": 1e-15},
        )

        w = np.asarray(res.x, dtype=float)
        w = project_to_box_simplex(w, lower, upper)

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
        raise RuntimeError("Max Sharpe optimization failed for all starting values.")

    return best_w


# =========================
# Output
# =========================
def save_weights_single_column(w: np.ndarray, path: Path) -> None:
    out = pd.DataFrame({"W": w})
    out.to_csv(path, index=False)


def print_weights(title: str, w: np.ndarray, asset_names: list[str]) -> None:
    print(f"\n{title}")
    tmp = pd.DataFrame({"Asset": asset_names, "W": w})
    print(tmp.to_string(index=False))
    print(f"sum(weights) = {w.sum():.12f}")


# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cov, asset_names = read_cov_matrix(COV_FILE)
    mu = read_means(MEAN_FILE)
    rf = 0.04

    # 10.1 Risk Parity, Normal Assumption
    w_10_1 = solve_risk_parity(cov)
    save_weights_single_column(w_10_1, OUT_10_1)
    print_weights("10.1 Risk Parity, Normal Assumption", w_10_1, asset_names)

    # 10.2 Risk Parity, Normal Assumption, x5 has 1/2 risk weight
    budget_10_2 = np.array([1.0, 1.0, 1.0, 1.0, 0.5], dtype=float)
    w_10_2 = solve_risk_parity(cov, budget=budget_10_2)
    save_weights_single_column(w_10_2, OUT_10_2)
    print_weights("10.2 Risk Parity, x5 has 1/2 risk weight", w_10_2, asset_names)

    # 10.3 Max Sharpe Ratio, normal assumption, w > 0
    w_10_3 = solve_max_sharpe(mu, cov, rf=rf, lower=1e-6, upper=1.0)
    save_weights_single_column(w_10_3, OUT_10_3)
    print_weights("10.3 Max Sharpe Ratio, w > 0", w_10_3, asset_names)

    # 10.4 Max Sharpe Ratio, normal assumption, 0.1 <= w <= 0.5
    w_10_4 = solve_max_sharpe(mu, cov, rf=rf, lower=0.1, upper=0.5)
    save_weights_single_column(w_10_4, OUT_10_4)
    print_weights("10.4 Max Sharpe Ratio, 0.1 <= w <= 0.5", w_10_4, asset_names)

    print("\nOutput files written to:")
    print(OUT_10_1)
    print(OUT_10_2)
    print(OUT_10_3)
    print(OUT_10_4)


if __name__ == "__main__":
    main()