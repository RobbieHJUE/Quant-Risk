from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Config:
    data_dir: str = "course/testfiles/data"
    out_dir: str = "HW04/out"
    seed: int = 100000
    n_sims: int = 5_000_000
    alpha_es: float = 0.05



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def must_exist(data_dir: Path, name: str) -> Path:
    p = data_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def read_csv_1col(path: Path, col: str = "x1") -> np.ndarray:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {path.name}, got columns={list(df.columns)}")
    x = df[col].to_numpy(dtype=float)
    return x.reshape(-1)


def two_col_es_df(es_abs: float, mean_return: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "ES Absolute": float(es_abs),
        "ES Diff from Mean": float(es_abs + mean_return),
    }])


def empirical_es_loss_from_returns(r: np.ndarray, alpha: float) -> float:
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return -float(tail.mean()) if tail.size > 0 else float("nan")


def es_loss_from_normal_fit(r: np.ndarray, alpha: float) -> float:
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    z = stats.norm.ppf(alpha)
    phi = stats.norm.pdf(z)
    cond_mean = mu - sigma * (phi / alpha)  
    return -float(cond_mean)         


def fit_student_t(r: np.ndarray) -> Tuple[float, float, float]:
    df, loc, scale = stats.t.fit(r)
    return float(df), float(loc), float(scale)


def es_loss_from_t_fit(r: np.ndarray, alpha: float) -> Tuple[float, float, float, float]:
    df, loc, scale = fit_student_t(r)
    if df <= 1:
        raise ValueError(f"Fitted df={df:.6f} <= 1, ES not defined.")
    c = stats.t.ppf(alpha, df)
    f = stats.t.pdf(c, df)
    F = stats.t.cdf(c, df)
    cond_mean_std = -((df + c**2) / (df - 1.0)) * (f / F) 
    cond_mean = loc + scale * cond_mean_std 
    es_loss = -float(cond_mean)
    return df, loc, scale, es_loss


def es_loss_from_t_simulation(df: float, loc: float, scale: float, alpha: float, n_sims: int, rng: np.random.Generator) -> float:
    r_sim = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims, random_state=rng)
    return empirical_es_loss_from_returns(np.asarray(r_sim, dtype=float), alpha)


def gaussian_copula_var_es_levels(
    portfolio_csv: Path,
    returns_csv: Path,
    alpha: float,
    n_sims: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    port = pd.read_csv(portfolio_csv)
    rets = pd.read_csv(returns_csv)

    stocks = [str(s) for s in port["Stock"].tolist()]

    marginals = {}
    for _, row in port.iterrows():
        s = str(row["Stock"])
        dist = str(row["Distribution"]).strip().lower()
        x = rets[s].to_numpy(dtype=float)

        if dist == "normal":
            mu = float(np.mean(x))
            sigma = float(np.std(x, ddof=1))
            marginals[s] = ("normal", (mu, sigma))
        elif dist in ("t", "student", "student-t", "student_t", "studentt"):
            df, loc, scale = stats.t.fit(x)
            df, loc, scale = float(df), float(loc), float(scale)
            marginals[s] = ("t", (df, loc, scale))
        else:
            raise ValueError(f"Unknown Distribution='{row['Distribution']}' for {s}")

    U = np.zeros((len(rets), len(stocks)), dtype=float)
    for j, s in enumerate(stocks):
        x = rets[s].to_numpy(dtype=float)
        ranks = stats.rankdata(x, method="average")
        U[:, j] = (ranks - 0.5) / len(x)

    Z = stats.norm.ppf(U)
    corr = np.corrcoef(Z, rowvar=False)
    L = np.linalg.cholesky(corr)

    z = rng.standard_normal(size=(n_sims, len(stocks))) @ L.T
    u = stats.norm.cdf(z)

    r_sim = np.zeros_like(u)
    for j, s in enumerate(stocks):
        kind, params = marginals[s]
        if kind == "normal":
            mu, sigma = params
            r_sim[:, j] = stats.norm.ppf(u[:, j], loc=mu, scale=sigma)
        else:
            df, loc, scale = params
            r_sim[:, j] = stats.t.ppf(u[:, j], df=df, loc=loc, scale=scale)

    # --- compute per-asset and total PnL/loss in dollars ---
    # initial values per asset
    v0_asset = {}
    for _, row in port.iterrows():
        s = str(row["Stock"])
        holding = float(row["Holding"])
        p0 = float(row["Starting Price"])
        v0_asset[s] = holding * p0
    V0_total = float(sum(v0_asset.values()))

    # simulated value change per asset
    pnl_asset = {}
    for j, s in enumerate(stocks):
        holding = float(port.loc[port["Stock"] == s, "Holding"].iloc[0])
        p0 = float(port.loc[port["Stock"] == s, "Starting Price"].iloc[0])
        p1 = p0 * (1.0 + r_sim[:, j])
        pnl_asset[s] = holding * (p1 - p0)

    pnl_total = np.zeros(n_sims, dtype=float)
    for s in stocks:
        pnl_total += pnl_asset[s]

    # helper: loss VaR/ES at 95% (alpha=0.05)
    def var_es_loss_from_pnl(pnl: np.ndarray) -> tuple[float, float]:
        loss = -pnl
        var = float(np.quantile(loss, 1.0 - alpha))
        es = float(loss[loss >= var].mean())
        return var, es

    rows = []
    for s in stocks:
        var, es = var_es_loss_from_pnl(pnl_asset[s])
        V0 = v0_asset[s]
        rows.append({
            "Stock": s,
            "VaR95": var,
            "ES95": es,
            "VaR95_Pct": var / V0,
            "ES95_Pct": es / V0,
        })

    varT, esT = var_es_loss_from_pnl(pnl_total)
    rows.append({
        "Stock": "Total",
        "VaR95": varT,
        "ES95": esT,
        "VaR95_Pct": varT / V0_total,
        "ES95_Pct": esT / V0_total,
    })

    return pd.DataFrame(rows)




def task_8_4(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    x = read_csv_1col(must_exist(data_dir, "test7_1.csv"), col="x1")
    es_loss = es_loss_from_normal_fit(x, cfg.alpha_es)
    df_out = two_col_es_df(es_loss, mean_return = np.mean(x))
    write_csv(out_dir / "testout_8.4.csv", df_out)
    return df_out


def task_8_5(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    x = read_csv_1col(must_exist(data_dir, "test7_2.csv"), col="x1")
    _, _, _, es_loss = es_loss_from_t_fit(x, cfg.alpha_es)
    df_out = two_col_es_df(es_loss, mean_return = np.mean(x))
    write_csv(out_dir / "testout_8.5.csv", df_out)
    return df_out


def task_8_6(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    x = read_csv_1col(must_exist(data_dir, "test7_2.csv"), col="x1")

    df_fit, loc, scale, es_analytic = es_loss_from_t_fit(x, cfg.alpha_es)
    es_sim = es_loss_from_t_simulation(df_fit, loc, scale, cfg.alpha_es, cfg.n_sims, rng)

    df_out = two_col_es_df(es_sim, mean_return = np.mean(x))
    write_csv(out_dir / "testout_8.6.csv", df_out)
    return df_out


def task_9_1(cfg: Config, data_dir: Path, out_dir: Path, rng: np.random.Generator) -> pd.DataFrame:
    df = gaussian_copula_var_es_levels(
        portfolio_csv=must_exist(data_dir, "test9_1_portfolio.csv"),
        returns_csv=must_exist(data_dir, "test9_1_returns.csv"),
        alpha=0.05,
        n_sims=cfg.n_sims,
        rng=rng,
    )
    write_csv(out_dir / "testout_9.1.csv", df)
    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir or Config.data_dir,
        out_dir=args.out_dir or Config.out_dir,
        seed=args.seed if args.seed is not None else Config.seed,
        n_sims=args.n_sims if args.n_sims is not None else Config.n_sims,
    )

    data_dir = Path(cfg.data_dir)
    out_dir = ensure_dir(cfg.out_dir)
    rng = np.random.default_rng(cfg.seed)

    results: Dict[str, Any] = {}
    results["8.4"] = task_8_4(cfg, data_dir, out_dir, rng).to_dict(orient="records")
    results["8.5"] = task_8_5(cfg, data_dir, out_dir, rng).to_dict(orient="records")
    results["8.6"] = task_8_6(cfg, data_dir, out_dir, rng).to_dict(orient="records")
    results["9.1"] = task_9_1(cfg, data_dir, out_dir, rng).to_dict(orient="records")

    if not args.quiet:
        print("Done. Outputs written to:", out_dir.resolve())
        for fn in ["testout_8.4.csv", "testout_8.5.csv", "testout_8.6.csv", "testout9_1.csv"]:
            p = out_dir / fn
            print(" -", fn, "OK" if p.exists() else "MISSING")


if __name__ == "__main__":
    main()