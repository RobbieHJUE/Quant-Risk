import os
import numpy as np
import pandas as pd
from scipy import stats  # Required for Part 8

# ================= Configuration =================
COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW03/out"

N_SIMS = 100_000
SEED = 100000
ALPHA = 0.05  # default alpha if file doesn't provide it


# ---------- IO helpers ----------
def read_matrix_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If first column is labels (e.g., var), use it as index
    if df.shape[1] >= 2 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        idx = df.iloc[:, 0].astype(str)
        mat = df.iloc[:, 1:].astype(float)
        mat.index = idx
        mat.columns = [str(c) for c in mat.columns]
        return mat
    return df.astype(float)


def write_matrix_csv(mat: pd.DataFrame, out_path: str) -> None:
    out_df = mat.copy()
    out_df.columns = [str(c) for c in out_df.columns]
    out_df.to_csv(out_path, index=False)


# ---------- Linear algebra helpers ----------
def proj_psd(A: np.ndarray) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    B = (V * w) @ V.T
    return 0.5 * (B + B.T)


def near_psd_cov_via_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cov -> corr using its diag, fix corr by eigen-clipping, then scale back.
    """
    S = cov.to_numpy(dtype=float)
    S = 0.5 * (S + S.T)

    v = np.diag(S)
    sd = np.sqrt(np.maximum(v, 0.0))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / denom
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    R_fixed = proj_psd(R)
    d = np.sqrt(np.diag(R_fixed))
    with np.errstate(divide="ignore", invalid="ignore"):
        Dinv = np.diag(1.0 / d)
        R_fixed = Dinv @ R_fixed @ Dinv
    R_fixed = 0.5 * (R_fixed + R_fixed.T)
    np.fill_diagonal(R_fixed, 1.0)

    D = np.diag(sd)
    S_fixed = D @ R_fixed @ D
    S_fixed = 0.5 * (S_fixed + S_fixed.T)

    return pd.DataFrame(S_fixed, index=cov.index, columns=cov.columns)


def higham_near_psd_with_fixed_diag(A: np.ndarray, diag_target: np.ndarray, tol: float = 1e-12, maxiter: int = 1000) -> np.ndarray:
    """
    Higham-style alternating projections:
      PSD projection + set diagonal to target
    """
    Y = A.copy()
    deltaS = np.zeros_like(A)
    normA = np.linalg.norm(A, ord="fro") + 1e-15

    for _ in range(maxiter):
        R = Y - deltaS
        X = proj_psd(R)
        deltaS = X - R

        Y_new = X.copy()
        np.fill_diagonal(Y_new, diag_target)

        if np.linalg.norm(Y_new - Y, ord="fro") / normA < tol:
            Y = Y_new
            break
        Y = Y_new

    X_final = proj_psd(Y)
    np.fill_diagonal(X_final, diag_target)
    return 0.5 * (X_final + X_final.T)


def higham_cov_via_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    cov -> corr, Higham-fix corr with diag=1, rescale back using original sd.
    """
    S = cov.to_numpy(dtype=float)
    S = 0.5 * (S + S.T)

    v = np.diag(S)
    sd = np.sqrt(np.maximum(v, 0.0))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / denom
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    R_fixed = higham_near_psd_with_fixed_diag(R, diag_target=np.ones(R.shape[0]), tol=1e-12, maxiter=2000)

    d = np.sqrt(np.diag(R_fixed))
    with np.errstate(divide="ignore", invalid="ignore"):
        Dinv = np.diag(1.0 / d)
        R_fixed = Dinv @ R_fixed @ Dinv
    R_fixed = 0.5 * (R_fixed + R_fixed.T)
    np.fill_diagonal(R_fixed, 1.0)

    D = np.diag(sd)
    S_fixed = D @ R_fixed @ D
    S_fixed = 0.5 * (S_fixed + S_fixed.T)

    return pd.DataFrame(S_fixed, index=cov.index, columns=cov.columns)


def chol_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    PSD Cholesky (no jitter). Returns lower-triangular L such that A ≈ L L^T.
    """
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        s = A[i, i] - np.dot(L[i, :i], L[i, :i])
        if s < 0 and abs(s) <= eps:
            s = 0.0
        if s < 0:
            raise RuntimeError(f"Matrix not PSD enough at diag {i}: {s}")
        L[i, i] = np.sqrt(s)

        if L[i, i] > 0:
            for j in range(i + 1, n):
                t = A[j, i] - np.dot(L[j, :i], L[i, :i])
                L[j, i] = t / L[i, i]

    return L


def simulate_cov_from_sigma(Sigma: pd.DataFrame, n_sims: int, seed: int) -> pd.DataFrame:
    S = Sigma.to_numpy(dtype=float)
    cols = list(Sigma.columns)

    try:
        L = np.linalg.cholesky(0.5 * (S + S.T))
    except np.linalg.LinAlgError:
        L = chol_psd(S)

    rs = np.random.RandomState(seed)
    Z = rs.standard_normal(size=(n_sims, S.shape[0]))
    X = Z @ L.T

    # keep your convention as-is (you said Part 5 is OK)
    Scov = np.cov(X, rowvar=False)  # ddof=0
    return pd.DataFrame(Scov, index=cols, columns=cols)


def pca_simulate_cov(Sigma: pd.DataFrame, explained: float, n_sims: int, seed: int) -> pd.DataFrame:
    S = Sigma.to_numpy(dtype=float)
    S = 0.5 * (S + S.T)
    cols = list(Sigma.columns)

    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w_pos = np.maximum(w, 0.0)
    total = float(np.sum(w_pos)) if np.sum(w_pos) > 0 else 1.0
    csum = np.cumsum(w_pos) / total
    k = int(np.searchsorted(csum, explained) + 1)

    Vk = V[:, :k]
    wk = w_pos[:k]
    B = Vk * np.sqrt(wk)  # n x k

    rs = np.random.RandomState(seed)
    Z = rs.standard_normal(size=(n_sims, k))
    X = Z @ B.T

    Scov = np.cov(X, rowvar=False, ddof=1)
    return pd.DataFrame(Scov, index=cols, columns=cols)


# ---------- Part 5 ----------
def run_part5():
    os.makedirs(OUT_DIR, exist_ok=True)

    sig_51 = read_matrix_csv(os.path.join(COURSE_DATA, "test5_1.csv"))
    cov_51 = simulate_cov_from_sigma(sig_51, n_sims=N_SIMS, seed=SEED)
    write_matrix_csv(cov_51, os.path.join(OUT_DIR, "testout_5.1.csv"))

    sig_52 = read_matrix_csv(os.path.join(COURSE_DATA, "test5_2.csv"))
    cov_52 = simulate_cov_from_sigma(sig_52, n_sims=N_SIMS, seed=SEED)
    write_matrix_csv(cov_52, os.path.join(OUT_DIR, "testout_5.2.csv"))

    sig_53_raw = read_matrix_csv(os.path.join(COURSE_DATA, "test5_3.csv"))
    sig_53 = near_psd_cov_via_corr(sig_53_raw)
    cov_53 = simulate_cov_from_sigma(sig_53, n_sims=N_SIMS, seed=SEED)
    write_matrix_csv(cov_53, os.path.join(OUT_DIR, "testout_5.3.csv"))

    sig_54 = higham_cov_via_corr(sig_53_raw)
    cov_54 = simulate_cov_from_sigma(sig_54, n_sims=N_SIMS, seed=SEED)
    write_matrix_csv(cov_54, os.path.join(OUT_DIR, "testout_5.4.csv"))

    cov_55 = pca_simulate_cov(sig_52, explained=0.99, n_sims=N_SIMS, seed=SEED)
    write_matrix_csv(cov_55, os.path.join(OUT_DIR, "testout_5.5.csv"))

    print("Part 5 wrote: testout_5.1.csv ... testout_5.5.csv")


# ---------- Part 8 (VaR) ----------
def read_series_and_alpha(path: str, default_alpha: float = ALPHA):
    """
    Returns (x, alpha):
      - alpha: use alpha/p/prob column if present, else default_alpha
      - x: prefer column 'x1', else first numeric column
    """
    df = pd.read_csv(path)

    alpha = float(default_alpha)
    for c in ["alpha", "p", "prob"]:
        if c in df.columns:
            alpha = float(df[c].iloc[0])
            break

    if "x1" in df.columns and np.issubdtype(df["x1"].dtype, np.number):
        x = df["x1"].to_numpy(dtype=float)
    else:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError(f"No numeric column found in {path}")
        x = df[num_cols[0]].to_numpy(dtype=float)

    x = x[~np.isnan(x)]
    return x, alpha


def var_metrics_from_quantile(q: float, mean_value: float) -> pd.DataFrame:
    """
    Output exactly two columns:
      - VaR Absolute = -q
      - VaR Diff from Mean = mean - q
    """
    return pd.DataFrame(
        {
            "VaR Absolute": [float(-q)],
            "VaR Diff from Mean": [float(mean_value - q)],
        }
    )


def run_part8():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 8.1 VaR from Normal Distribution
    x1, a1 = read_series_and_alpha(os.path.join(COURSE_DATA, "test7_1.csv"))
    mu1 = float(np.mean(x1))
    sd1 = float(np.std(x1, ddof=1))
    q1 = float(mu1 + sd1 * stats.norm.ppf(a1))
    out81 = var_metrics_from_quantile(q1, mu1)
    out81.to_csv(os.path.join(OUT_DIR, "testout8_1.csv"), index=False)

    # 8.2 VaR from T Distribution
    x2, a2 = read_series_and_alpha(os.path.join(COURSE_DATA, "test7_2.csv"))
    nu, loc, scale = stats.t.fit(x2)
    q2 = float(stats.t.ppf(a2, df=nu, loc=loc, scale=scale))
    out82 = var_metrics_from_quantile(q2, float(loc))
    out82.to_csv(os.path.join(OUT_DIR, "testout8_2.csv"), index=False)

    # 8.3 VaR from Simulation (compare to 8.2)
    rs = np.random.RandomState(SEED)
    sim = stats.t.rvs(df=nu, loc=loc, scale=scale, size=N_SIMS, random_state=rs)
    q3 = float(np.quantile(sim, a2))
    out83 = var_metrics_from_quantile(q3, float(loc))
    out83.to_csv(os.path.join(OUT_DIR, "testout8_3.csv"), index=False)

    print("Part 8 wrote: testout8_1.csv ... testout8_3.csv")


def main():
    run_part5()
    run_part8()
    print(f"All Tasks Done. Files saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
