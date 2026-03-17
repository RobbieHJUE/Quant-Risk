# run_all.py
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SQRT_2PI = math.sqrt(2.0 * math.pi)

def time_to_maturity_from_row(row):
    days = safe_float(row["DaysToMaturity"])
    day_per_year = safe_float(row["DayPerYear"])

    if day_per_year <= 0:
        raise ValueError(
            f"Invalid DayPerYear={day_per_year} for ID={row.get('ID', 'UNKNOWN')}. "
            f"Check CSV column names and values."
        )

    return days / day_per_year

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def parse_option_type(x: str) -> str:
    s = str(x).strip().lower()
    if s.startswith("c"):
        return "call"
    elif s.startswith("p"):
        return "put"
    raise ValueError(f"Unknown option type: {x}")


def safe_float(x, default=0.0):
    if pd.isna(x):
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return default
    return float(s)


def ensure_positive(x: float, eps: float = 1e-12) -> float:
    return max(float(x), eps)


# =========================================================
# 12.1 European Options GBSM including Greeks
# =========================================================
def gbsm_price_greeks(S, K, T, r, q, sigma, option_type):
    """
    Generalized Black-Scholes-Merton with continuous dividend yield q.
    Returns: price, delta, gamma, vega, theta, rho, carry_rho
    """
    option_type = parse_option_type(option_type)

    S = ensure_positive(S)
    K = ensure_positive(K)
    T = ensure_positive(T)
    sigma = ensure_positive(sigma)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "call":
        price = S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * sqrtT)
            - r * K * math.exp(-r * T) * norm_cdf(d2)
            + q * S * math.exp(-q * T) * norm_cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
        carry_rho = T * S * math.exp(-q * T) * norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1.0)
        theta = (
            -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * sqrtT)
            + r * K * math.exp(-r * T) * norm_cdf(-d2)
            - q * S * math.exp(-q * T) * norm_cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
        carry_rho = -T * S * math.exp(-q * T) * norm_cdf(-d1)

    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * norm_pdf(d1) * sqrtT

    return {
        "Value": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Rho": rho, 
        "Theta": theta,
    }


# =========================================================
# 12.2 American Options with continuous Dividends including Greeks
# =========================================================

def american_binomial_continuous_price_b(
    S, K, T, r, b, sigma, option_type, steps=400
):
    option_type = parse_option_type(option_type)

    S = ensure_positive(S)
    K = ensure_positive(K)
    T = ensure_positive(T)
    sigma = ensure_positive(sigma)

    if T <= 1e-12:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    N = max(int(steps), 3)
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)

    # IMPORTANT: probability uses b, not r directly
    p = (math.exp(b * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    vals = np.zeros(N + 1)
    for j in range(N + 1):
        ST = S * (u ** j) * (d ** (N - j))
        if option_type == "call":
            vals[j] = max(ST - K, 0.0)
        else:
            vals[j] = max(K - ST, 0.0)

    for i in range(N - 1, -1, -1):
        new_vals = np.zeros(i + 1)
        for j in range(i + 1):
            stock = S * (u ** j) * (d ** (i - j))
            continuation = disc * (p * vals[j + 1] + (1.0 - p) * vals[j])
            if option_type == "call":
                exercise = max(stock - K, 0.0)
            else:
                exercise = max(K - stock, 0.0)
            new_vals[j] = max(exercise, continuation)
        vals = new_vals

    return float(vals[0])


def american_continuous_dividend_price_greeks(
    S, K, T, r, q, sigma, option_type, steps=400
):
    """
    For dividend-paying stock, use b = r - q.
    Rho: bump r while holding b fixed.
    """
    option_type = parse_option_type(option_type)

    S = ensure_positive(S)
    K = ensure_positive(K)
    T = ensure_positive(T)
    sigma = ensure_positive(sigma)

    b = r - q

    if T <= 1e-12:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return {
            "Price": intrinsic,
            "Delta": 1.0 if (option_type == "call" and S > K) else (-1.0 if (option_type == "put" and S < K) else 0.0),
            "Gamma": 0.0,
            "Vega": 0.0,
            "Rho": 0.0,
            "Theta": 0.0,
        }

    N = max(int(steps), 3)
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(b * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    vals = np.zeros(N + 1)
    for j in range(N + 1):
        ST = S * (u ** j) * (d ** (N - j))
        if option_type == "call":
            vals[j] = max(ST - K, 0.0)
        else:
            vals[j] = max(K - ST, 0.0)

    level_2_vals = None
    level_1_vals = None

    for i in range(N - 1, -1, -1):
        new_vals = np.zeros(i + 1)
        for j in range(i + 1):
            stock = S * (u ** j) * (d ** (i - j))
            continuation = disc * (p * vals[j + 1] + (1.0 - p) * vals[j])
            if option_type == "call":
                exercise = max(stock - K, 0.0)
            else:
                exercise = max(K - stock, 0.0)
            new_vals[j] = max(exercise, continuation)

        if i == 2:
            level_2_vals = new_vals.copy()
        if i == 1:
            level_1_vals = new_vals.copy()

        vals = new_vals

    price = float(vals[0])

    # Delta
    Su = S * u
    Sd = S * d
    Vu = float(level_1_vals[1])
    Vd = float(level_1_vals[0])
    delta = (Vu - Vd) / (Su - Sd)

    # Gamma
    Suu = S * u * u
    Sud = S * u * d
    Sdd = S * d * d
    Vuu = float(level_2_vals[2])
    Vud = float(level_2_vals[1])
    Vdd = float(level_2_vals[0])

    delta_up = (Vuu - Vud) / (Suu - Sud)
    delta_down = (Vud - Vdd) / (Sud - Sdd)
    gamma = (delta_up - delta_down) / ((Suu - Sdd) / 2.0)

    # Theta
    theta = (Vud - price) / (2.0 * dt)

    # Vega: bump sigma, keep r and b fixed
    hV = 1e-4
    upV = american_binomial_continuous_price_b(
        S, K, T, r, b, sigma + hV, option_type, steps=steps
    )
    dnV = american_binomial_continuous_price_b(
        S, K, T, r, b, max(sigma - hV, 1e-8), option_type, steps=steps
    )
    vega = (upV - dnV) / (2.0 * hV)

    # Rho: bump r, HOLD b FIXED
    hR = 1e-4
    upR = american_binomial_continuous_price_b(
        S, K, T, r + hR, b, sigma, option_type, steps=steps
    )
    dnR = american_binomial_continuous_price_b(
        S, K, T, r - hR, b, sigma, option_type, steps=steps
    )
    rho = (upR - dnR) / (2.0 * hR)

    return {
        "Value": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Rho": rho,
        "Theta": theta,
    }


# =========================================================
# 12.3 American Options with discrete Dividends
# =========================================================
def parse_dividend_list(x):
    """
    Input examples:
      "75,150"
      ".01,.01"
      75
      NaN
    """
    if pd.isna(x):
        return []
    if isinstance(x, (int, float, np.integer, np.floating)):
        return [float(x)]
    s = str(x).strip()
    if s == "":
        return []
    return [float(v.strip()) for v in s.split(",") if v.strip() != ""]


def american_binomial_discrete_dividend_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type,
    dividend_times=None,
    dividend_amounts=None,
    steps=500,
):
    """
    American option with discrete cash dividends using a step-by-step binomial tree.

    Assumption:
    - dividend_times are in YEARS from today
    - dividend_amounts are cash dividends
    - dividends with time <= T are applied
    """
    option_type = parse_option_type(option_type)

    S = ensure_positive(S)
    K = ensure_positive(K)
    sigma = ensure_positive(sigma)
    T = ensure_positive(T)

    dividend_times = dividend_times or []
    dividend_amounts = dividend_amounts or []

    # keep only dividends before maturity
    divs = [(t, a) for t, a in zip(dividend_times, dividend_amounts) if 0.0 < t <= T]
    divs.sort(key=lambda x: x[0])

    N = max(int(steps), 1)
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    # map dividend dates to nearest step
    div_by_step = {}
    for t, amt in divs:
        step = int(round(t / dt))
        step = min(max(step, 1), N)
        div_by_step[step] = div_by_step.get(step, 0.0) + amt

    # forward build stock tree
    stock_tree = [[S]]
    for i in range(1, N + 1):
        prev = stock_tree[-1]
        curr = [0.0] * (i + 1)

        # first node (all downs)
        curr[0] = prev[0] * d
        # interior
        for j in range(1, i):
            # recombining tree value from either parent; these are numerically close
            from_down_parent = prev[j] * d
            from_up_parent = prev[j - 1] * u
            curr[j] = 0.5 * (from_down_parent + from_up_parent)
        # last node (all ups)
        curr[i] = prev[i - 1] * u

        # apply cash dividend right after arriving at this step
        if i in div_by_step:
            D = div_by_step[i]
            curr = [max(x - D, 1e-12) for x in curr]

        stock_tree.append(curr)

    # terminal payoff
    values = [0.0] * (N + 1)
    for j in range(N + 1):
        ST = stock_tree[N][j]
        if option_type == "call":
            values[j] = max(ST - K, 0.0)
        else:
            values[j] = max(K - ST, 0.0)

    # backward induction
    for i in range(N - 1, -1, -1):
        new_vals = [0.0] * (i + 1)
        for j in range(i + 1):
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            stock = stock_tree[i][j]
            if option_type == "call":
                exercise = max(stock - K, 0.0)
            else:
                exercise = max(K - stock, 0.0)
            new_vals[j] = max(exercise, continuation)
        values = new_vals

    return float(values[0])


# =========================================================
# File runners
# =========================================================
def run_12_1(input_path: Path, output_path: Path):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = df[df["ID"].notna()]
    df = df[df["DayPerYear"].notna()]
    df = df[df["DayPerYear"] != 0]
    rows = []


    for _, row in df.iterrows():
        option_type = row["Option Type"]
        S = safe_float(row["Underlying"])
        K = safe_float(row["Strike"])
        T = time_to_maturity_from_row(row)
        r = safe_float(row["RiskFreeRate"])
        q = safe_float(row["DividendRate"])
        sigma = safe_float(row["ImpliedVol"])

        result = gbsm_price_greeks(S, K, T, r, q, sigma, option_type)
        rows.append(
            {
                "ID": row["ID"],
                "Value": result["Value"],
                "Delta": result["Delta"],
                "Gamma": result["Gamma"],
                "Vega": result["Vega"],
                "Rho": result["Rho"], 
                "Theta": result["Theta"],
                            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)


def run_12_2(input_path: Path, output_path: Path, steps=400):
    df = pd.read_csv(input_path)
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = df[df["ID"].notna()]
    df = df[df["DayPerYear"].notna()]
    df = df[df["DayPerYear"] != 0]
    rows = []

    for _, row in df.iterrows():
        option_type = row["Option Type"]
        S = safe_float(row["Underlying"])
        K = safe_float(row["Strike"])
        T = time_to_maturity_from_row(row)
        r = safe_float(row["RiskFreeRate"])
        q = safe_float(row["DividendRate"])
        sigma = safe_float(row["ImpliedVol"])

        result = american_continuous_dividend_price_greeks(
            S, K, T, r, q, sigma, option_type, steps=steps
        )
        rows.append(
            {
                "ID": row["ID"],
                "Value": result["Value"],
                "Delta": result["Delta"],
                "Gamma": result["Gamma"],
                "Vega": result["Vega"],
                "Rho": result["Rho"],
                "Theta": result["Theta"],            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)


def run_12_3(input_path: Path, output_path: Path, steps=500):
    df = pd.read_csv(input_path)
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    df = df[df["ID"].notna()]
    df = df[df["DayPerYear"].notna()]
    df = df[df["DayPerYear"] != 0]
    rows = []

    for _, row in df.iterrows():
        option_type = row["Option Type"]
        S = safe_float(row["Underlying"])
        K = safe_float(row["Strike"])
        T = time_to_maturity_from_row(row)
        r = safe_float(row["RiskFreeRate"])
        sigma = safe_float(row["ImpliedVol"])

        div_dates_days = parse_dividend_list(row["DividendDates"])
        div_amts = parse_dividend_list(row["DividendAmts"])
        day_per_year = safe_float(row["DayPerYear"])

        div_times = [d / day_per_year for d in div_dates_days]

        price = american_binomial_discrete_dividend_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            dividend_times=div_times,
            dividend_amounts=div_amts,
            steps=steps,
        )

        rows.append(
            {
                "ID": row["ID"],
                "Price": price,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)


def main():
    data_dir = Path("course/testfiles/data")
    out_dir = Path("HW05/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_121 = data_dir / "test12_1.csv"
    input_123 = data_dir / "test12_3.csv"

    output_121 = out_dir / "testout12_1.csv"
    output_122 = out_dir / "testout12_2.csv"
    output_123 = out_dir / "testout12_3.csv"

    run_12_1(input_121, output_121)
    run_12_2(input_121, output_122)
    run_12_3(input_123, output_123)

    print("Done.")
    print(f"Wrote: {output_121}")
    print(f"Wrote: {output_122}")
    print(f"Wrote: {output_123}")


if __name__ == "__main__":
    main()