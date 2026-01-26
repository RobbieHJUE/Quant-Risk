import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW01/out"


def fit_normal(path_csv: str) -> pd.DataFrame:
    x = pd.read_csv(path_csv)["x1"].to_numpy()
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    return pd.DataFrame([{"mu": mu, "sigma": sigma}])


def fit_t_dist(path_csv: str) -> pd.DataFrame:
    x = pd.read_csv(path_csv)["x1"].to_numpy()
    nu, mu, sigma = stats.t.fit(x)
    return pd.DataFrame([{"mu": float(mu), "sigma": float(sigma), "nu": float(nu)}])


def fit_t_regression(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    X = df[["x1", "x2", "x3"]].to_numpy()
    y = df["y"].to_numpy()
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])

    beta0 = np.linalg.lstsq(X1, y, rcond=None)[0]
    resid = y - X1 @ beta0
    sigma0 = float(np.std(resid, ddof=1))
    nu0 = 5.0

    def nll(theta):
        beta = theta[:4]
        sigma = np.exp(theta[4])
        nu = np.exp(theta[5]) + 1e-12
        r = y - X1 @ beta
        return -np.sum(stats.t.logpdf(r, df=nu, loc=0.0, scale=sigma))

    theta0 = np.concatenate([beta0, [np.log(sigma0), np.log(nu0)]])
    res = minimize(nll, theta0, method="Nelder-Mead",
                   options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12})
    if not res.success:
        raise RuntimeError(res.message)

    beta = res.x[:4]
    sigma = float(np.exp(res.x[4]))
    nu = float(np.exp(res.x[5]) + 1e-12)

    return pd.DataFrame([{
        "mu": 0.0,
        "sigma": sigma,
        "nu": nu,
        "Alpha": float(beta[0]),
        "B1": float(beta[1]),
        "B2": float(beta[2]),
        "B3": float(beta[3]),
    }])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fit_normal(f"{COURSE_DATA}/test7_1.csv").to_csv(f"{OUT_DIR}/testout_7.1.csv", index=False)
    fit_t_dist(f"{COURSE_DATA}/test7_2.csv").to_csv(f"{OUT_DIR}/testout_7.2.csv", index=False)
    fit_t_regression(f"{COURSE_DATA}/test7_3.csv").to_csv(f"{OUT_DIR}/testout_7.3.csv", index=False)

    print("Wrote outputs to:", OUT_DIR)
    print(os.listdir(OUT_DIR))


if __name__ == "__main__":
    main()
