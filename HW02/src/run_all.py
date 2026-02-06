import os
from typing import Optional

import numpy as np
import pandas as pd

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW02/out"


# I/O helpers
def write_matrix_csv(mat: pd.DataFrame, out_path: str, label_col: str = "var") -> None:
    """Write a labeled square matrix to CSV with an explicit index/label column."""
    out_df = mat.copy()
    out_df.insert(0, label_col, out_df.index.astype(str))
    out_df.to_csv(out_path, index=False)


def read_matrix_csv(path: str) -> pd.DataFrame:
    """
    Read a square matrix CSV.
    Supports either:
      - first column is a non-numeric label column (e.g. 'var')
      - pure numeric matrix (no label column)
    """
    df = pd.read_csv(path)

    has_label_col = df.shape[1] >= 2 and not np.issubdtype(df.iloc[:, 0].dtype, np.number)
    if has_label_col:
        idx = df.iloc[:, 0].astype(str)
        mat = df.iloc[:, 1:].astype(float)
        mat.index = idx
        return mat

    return df.astype(float)


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns only."""
    return df.select_dtypes(include=[np.number]).copy()

# Pairwise covariance/correlation
def pairwise_cov(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise covariance: cov(i,j) uses rows where both i and j are present."""
    cols = list(df.columns)
    p = len(cols)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    # Diagonal: sample variance on each column's non-missing values
    for i, ci in enumerate(cols):
        x = df[ci].dropna().to_numpy()
        out.iloc[i, i] = float(np.var(x, ddof=1)) if x.size >= 2 else np.nan

    # Off-diagonal: sample covariance on complete pairs
    for i in range(p):
        for j in range(i + 1, p):
            both = df[[cols[i], cols[j]]].dropna()
            if len(both) >= 2:
                cov_ij = float(np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)[0, 1])
            else:
                cov_ij = np.nan
            out.iloc[i, j] = cov_ij
            out.iloc[j, i] = cov_ij

    return out


def pairwise_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation: corr(i,j) uses rows where both i and j are present."""
    cols = list(df.columns)
    p = len(cols)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    # Diagonal: 1 if sample std>0, else NaN
    for i, ci in enumerate(cols):
        x = df[ci].dropna().to_numpy()
        out.iloc[i, i] = 1.0 if (x.size >= 2 and np.std(x, ddof=1) > 0) else np.nan

    # Off-diagonal: corr on complete pairs if both have nonzero variance
    for i in range(p):
        for j in range(i + 1, p):
            both = df[[cols[i], cols[j]]].dropna()
            if len(both) >= 2:
                a = both.iloc[:, 0].to_numpy()
                b = both.iloc[:, 1].to_numpy()
                sa = np.std(a, ddof=1)
                sb = np.std(b, ddof=1)
                corr_ij = float(np.corrcoef(a, b)[0, 1]) if (sa > 0 and sb > 0) else np.nan
            else:
                corr_ij = np.nan
            out.iloc[i, j] = corr_ij
            out.iloc[j, i] = corr_ij

    return out


# Exponentially weighted covariance/correlation
def ew_weights(n: int, lam: float) -> np.ndarray:
    """
    Exponentially decaying weights for rows 0..n-1 (n-1 is most recent).
    w_t ∝ (1-lam) * lam^(n-1-t), normalized to sum to 1.
    """
    w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def ew_cov(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Exponentially weighted covariance using a weighted mean and weighted outer products."""
    cols = list(df.columns)
    X = df.to_numpy(dtype=float)
    n = X.shape[0]
    w = ew_weights(n, lam)

    mu = np.sum(X * w[:, None], axis=0)
    Xc = X - mu[None, :]
    S = (Xc * w[:, None]).T @ Xc

    return pd.DataFrame(S, index=cols, columns=cols)


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to correlation, handling zero/NaN variance safely."""
    cols = list(cov.columns)
    v = np.diag(cov.to_numpy(dtype=float))
    sd = np.sqrt(v)

    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = cov.to_numpy(dtype=float) / denom

    R = 0.5 * (R + R.T)

    for i in range(len(cols)):
        R[i, i] = 1.0 if (np.isfinite(sd[i]) and sd[i] > 0) else np.nan

    return pd.DataFrame(R, index=cols, columns=cols)


# Near-PSD projections
def proj_psd(A: np.ndarray) -> np.ndarray:
    """Project to the PSD cone via eigenvalue clipping."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    B = (V * w) @ V.T
    return 0.5 * (B + B.T)


def near_psd_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalue-clipped PSD approximation of a covariance matrix."""
    B = proj_psd(cov.to_numpy(dtype=float))
    return pd.DataFrame(B, index=cov.index, columns=cov.columns)


def near_psd_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """Eigenvalue-clipped PSD approximation of a correlation matrix (forces diag=1)."""
    B = proj_psd(corr.to_numpy(dtype=float))
    d = np.sqrt(np.diag(B))

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.diag(1.0 / d)
        C = S @ B @ S

    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=corr.index, columns=corr.columns)


def higham_near_psd_with_fixed_diag(
    A: np.ndarray,
    diag_target: np.ndarray,
    tol: float = 1e-12,
    maxiter: int = 1000,
) -> np.ndarray:
    """
    Higham-style alternating projections:
      1) project to PSD
      2) reset diagonal to diag_target
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


def higham_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """Higham-style near-PSD covariance with diagonal fixed to original variances."""
    A = cov.to_numpy(dtype=float)
    diag_target = np.diag(A).copy()
    B = higham_near_psd_with_fixed_diag(A, diag_target=diag_target)
    return pd.DataFrame(B, index=cov.index, columns=cov.columns)


def higham_corr(corr: pd.DataFrame) -> pd.DataFrame:
    """Higham-style near-PSD correlation with diagonal fixed to ones."""
    A = corr.to_numpy(dtype=float)
    diag_target = np.ones(A.shape[0], dtype=float)
    B = higham_near_psd_with_fixed_diag(A, diag_target=diag_target)

    d = np.sqrt(np.diag(B))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.diag(1.0 / d)
        C = S @ B @ S

    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=corr.index, columns=corr.columns)


# PSD Cholesky (no jitter; safe for semidefinite)
def chol_psd(A: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Lower-triangular L such that A ≈ L L^T for PSD matrices (incl. semidefinite)."""
    M = A.to_numpy(dtype=float)
    M = 0.5 * (M + M.T)
    n = M.shape[0]
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        s = M[i, i] - np.dot(L[i, :i], L[i, :i])

        if s < 0 and abs(s) <= eps:
            s = 0.0
        if s < 0:
            raise RuntimeError(f"Matrix not PSD at diagonal {i}: {s}")

        L[i, i] = np.sqrt(s)

        if L[i, i] > 0:
            for j in range(i + 1, n):
                t = M[j, i] - np.dot(L[j, :i], L[i, :i])
                L[j, i] = t / L[i, i]

    return pd.DataFrame(L, index=A.index, columns=A.columns)


# Parts run
def run_part1() -> None:
    inp = os.path.join(COURSE_DATA, "test1.csv")
    df = numeric_only(pd.read_csv(inp))

    df_cc = df.dropna(axis=0, how="any")
    cov_cc = df_cc.cov(ddof=1)
    write_matrix_csv(cov_cc, os.path.join(OUT_DIR, "testout_1.1.csv"))

    corr_cc = df_cc.corr()
    write_matrix_csv(corr_cc, os.path.join(OUT_DIR, "testout_1.2.csv"))

    cov_pw = pairwise_cov(df)
    write_matrix_csv(cov_pw, os.path.join(OUT_DIR, "testout_1.3.csv"))

    corr_pw = pairwise_corr(df)
    write_matrix_csv(corr_pw, os.path.join(OUT_DIR, "testout_1.4.csv"))

    print("Part 1 wrote: testout_1.1.csv ... testout_1.4.csv")


def run_part2() -> None:
    inp = os.path.join(COURSE_DATA, "test2.csv")
    df = numeric_only(pd.read_csv(inp))

    cov_097 = ew_cov(df, lam=0.97)
    write_matrix_csv(cov_097, os.path.join(OUT_DIR, "testout_2.1.csv"))

    cov_094 = ew_cov(df, lam=0.94)
    corr_094 = cov_to_corr(cov_094)
    write_matrix_csv(corr_094, os.path.join(OUT_DIR, "testout_2.2.csv"))

    var_097 = np.diag(cov_097.to_numpy(dtype=float))
    D = np.diag(np.sqrt(var_097))
    cov_23 = D @ corr_094.to_numpy(dtype=float) @ D
    cov_23 = pd.DataFrame(cov_23, index=cov_097.index, columns=cov_097.columns)
    write_matrix_csv(cov_23, os.path.join(OUT_DIR, "testout_2.3.csv"))

    print("Part 2 wrote: testout_2.1.csv ... testout_2.3.csv")


def _cov_to_corr_from_own_diag(cov: pd.DataFrame) -> pd.DataFrame:
    """Correlation computed from a covariance matrix using its own diagonal."""
    v = np.diag(cov.to_numpy(dtype=float))
    sd = np.sqrt(v)
    denom = np.outer(sd, sd)

    with np.errstate(divide="ignore", invalid="ignore"):
        C = cov.to_numpy(dtype=float) / denom

    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=cov.index, columns=cov.columns)


def run_part3() -> None:
    cov = read_matrix_csv(os.path.join(OUT_DIR, "testout_1.3.csv"))
    corr = read_matrix_csv(os.path.join(OUT_DIR, "testout_1.4.csv"))

    corr_from_cov = _cov_to_corr_from_own_diag(cov)
    sd = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    D = np.diag(sd)

    # 3.1: near-PSD covariance via near-PSD correlation then rescale
    corr_npsd_for_cov = near_psd_corr(corr_from_cov)
    cov_31 = D @ corr_npsd_for_cov.to_numpy(dtype=float) @ D
    cov_31 = pd.DataFrame(cov_31, index=cov.index, columns=cov.columns)
    write_matrix_csv(cov_31, os.path.join(OUT_DIR, "testout_3.1.csv"))

    # 3.2: near-PSD correlation
    corr_npsd = near_psd_corr(corr)
    write_matrix_csv(corr_npsd, os.path.join(OUT_DIR, "testout_3.2.csv"))

    # 3.3: Higham covariance via Higham correlation then rescale
    corr_higham_for_cov = higham_corr(corr_from_cov)
    cov_33 = D @ corr_higham_for_cov.to_numpy(dtype=float) @ D
    cov_33 = pd.DataFrame(cov_33, index=cov.index, columns=cov.columns)
    write_matrix_csv(cov_33, os.path.join(OUT_DIR, "testout_3.3.csv"))

    # 3.4: Higham correlation
    corr_higham = higham_corr(corr)
    write_matrix_csv(corr_higham, os.path.join(OUT_DIR, "testout_3.4.csv"))

    print("Part 3 wrote: testout_3.1.csv ... testout_3.4.csv")


def run_part4() -> None:
    cov31 = read_matrix_csv(os.path.join(OUT_DIR, "testout_3.1.csv"))
    L = chol_psd(cov31)
    write_matrix_csv(L, os.path.join(OUT_DIR, "testout_4.1.csv"))
    print("Part 4 wrote: testout_4.1.csv")


def _detect_date_col(df: pd.DataFrame) -> str:
    """Pick a likely date column; fall back to first column."""
    for c in df.columns:
        if str(c).lower() in {"date", "datetime", "time"}:
            return c
    return df.columns[0]


def run_part6() -> None:
    inp = os.path.join(COURSE_DATA, "test6.csv")
    df = pd.read_csv(inp)

    date_col = _detect_date_col(df)
    dates = df[[date_col]].iloc[1:].reset_index(drop=True)

    px = numeric_only(df)
    ar = px.pct_change().iloc[1:].reset_index(drop=True)
    lr = np.log(px).diff().iloc[1:].reset_index(drop=True)

    pd.concat([dates, ar], axis=1).to_csv(os.path.join(OUT_DIR, "testout6_1.csv"), index=False)
    pd.concat([dates, lr], axis=1).to_csv(os.path.join(OUT_DIR, "testout6_2.csv"), index=False)

    print("Part 6 wrote: testout6_1.csv and testout6_2.csv")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    run_part1()
    run_part2()
    run_part3()
    run_part4()
    run_part6()

    print("DONE. Outputs in:", OUT_DIR)
    print("Files:", os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
