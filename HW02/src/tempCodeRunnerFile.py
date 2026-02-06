import os
import numpy as np
import pandas as pd

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW02/out"


def write_matrix_csv(mat: pd.DataFrame, out_path: str) -> None:
    """
    Write a square matrix with clear row/col labels.
    Output format:
      var, x1, x2, ...
      x1, ...
      x2, ...
    """
    out_df = mat.copy()
    out_df.insert(0, "var", out_df.index)
    out_df.to_csv(out_path, index=False)


def pairwise_cov(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    p = len(cols)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i in range(p):
        xi = df[cols[i]]
        # diagonal: variance using all non-missing xi
        x = xi.dropna().to_numpy()
        if x.size >= 2:
            out.iloc[i, i] = float(np.var(x, ddof=1))
        else:
            out.iloc[i, i] = np.nan

        for j in range(i + 1, p):
            xj = df[cols[j]]
            both = pd.concat([xi, xj], axis=1).dropna()
            if len(both) >= 2:
                cov_ij = float(np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)[0, 1])
            else:
                cov_ij = np.nan
            out.iloc[i, j] = cov_ij
            out.iloc[j, i] = cov_ij

    return out


def pairwise_corr(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    p = len(cols)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i in range(p):
        xi = df[cols[i]].dropna().to_numpy()
        # diagonal: corr(x,x)=1 if there is variance, else NaN
        if xi.size >= 2 and np.std(xi, ddof=1) > 0:
            out.iloc[i, i] = 1.0
        else:
            out.iloc[i, i] = np.nan

        for j in range(i + 1, p):
            both = df[[cols[i], cols[j]]].dropna()
            if len(both) >= 2:
                a = both.iloc[:, 0].to_numpy()
                b = both.iloc[:, 1].to_numpy()
                sa = np.std(a, ddof=1)
                sb = np.std(b, ddof=1)
                if sa > 0 and sb > 0:
                    corr_ij = float(np.corrcoef(a, b)[0, 1])
                else:
                    corr_ij = np.nan
            else:
                corr_ij = np.nan

            out.iloc[i, j] = corr_ij
            out.iloc[j, i] = corr_ij

    return out


def run_part1():
    inp = os.path.join(COURSE_DATA, "test1.csv")
    df = pd.read_csv(inp)

    # 如果 test1.csv 里有非数值列（理论上不该有），先只保留数值列
    df = df.select_dtypes(include=[np.number])

    # 1.1 Covariance — skip missing rows (complete-case)
    df_cc = df.dropna(axis=0, how="any")
    cov_cc = df_cc.cov(ddof=1)
    write_matrix_csv(cov_cc, os.path.join(OUT_DIR, "testout_1.1.csv"))

    # 1.2 Correlation — skip missing rows (complete-case)
    corr_cc = df_cc.corr()
    write_matrix_csv(corr_cc, os.path.join(OUT_DIR, "testout_1.2.csv"))

    # 1.3 Covariance — pairwise
    cov_pw = pairwise_cov(df)
    write_matrix_csv(cov_pw, os.path.join(OUT_DIR, "testout_1.3.csv"))

    # 1.4 Correlation — pairwise
    corr_pw = pairwise_corr(df)
    write_matrix_csv(corr_pw, os.path.join(OUT_DIR, "testout_1.4.csv"))

    print("Part 1 wrote: testout_1.1.csv ... testout_1.4.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Part 1 ----
    run_part1()

    print("DONE. Outputs in:", OUT_DIR)
    print("Files:", os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
