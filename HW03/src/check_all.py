import os
import numpy as np
import pandas as pd

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW03/out"

TOL = 1e-3


def _read_numeric_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path)

    if df.shape[1] >= 2:
        first_name = str(df.columns[0]).strip().lower()
        if first_name == "var":
            df = df.iloc[:, 1:]
        elif not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df = df.iloc[:, 1:]

    return df.to_numpy(dtype=float)


def _read_scalar_var(path: str) -> float:
    df = pd.read_csv(path)
    if "var" in df.columns:
        return float(df["var"].iloc[0])
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return float(df[c].iloc[0])

    raise ValueError(f"No numeric value found in {path}")


def check_matrix(expected_path: str, got_path: str, tol: float = TOL) -> float:
    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Expected file not found: {expected_path}")
    if not os.path.exists(got_path):
        raise FileNotFoundError(f"Your output file not found: {got_path}")

    A = _read_numeric_matrix(expected_path)
    B = _read_numeric_matrix(got_path)

    if A.shape != B.shape:
        raise AssertionError(f"Shape mismatch: expected {A.shape}, got {B.shape}")

    diff = np.abs(A - B)
    maxdiff = float(np.max(diff)) if diff.size else 0.0

    if maxdiff > tol:
        i, j = np.argwhere(diff > tol)[0]
        raise AssertionError(
            f"Value mismatch (maxdiff={maxdiff:.3e}) at [{i},{j}]: expected {A[i,j]}, got {B[i,j]}"
        )

    return maxdiff


def check_scalar(expected_path: str, got_path: str, tol: float = TOL) -> float:
    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Expected file not found: {expected_path}")
    if not os.path.exists(got_path):
        raise FileNotFoundError(f"Your output file not found: {got_path}")

    a = _read_scalar_var(expected_path)
    b = _read_scalar_var(got_path)
    d = float(abs(a - b))

    if d > tol:
        raise AssertionError(f"Value mismatch: expected {a}, got {b} (absdiff={d:.3e})")

    return d


def main():
    matrix_tests = ["5.1", "5.2", "5.3", "5.4", "5.5"]
    scalar_tests = ["8.1", "8.2", "8.3"]

    diffs = {}
    all_pass = True

    for t in matrix_tests:
        exp = os.path.join(COURSE_DATA, f"testout_{t}.csv")
        got = os.path.join(OUT_DIR, f"testout_{t}.csv")
        try:
            d = check_matrix(exp, got, tol=TOL)
            diffs[t] = d
            print(f"PASS {t}  maxdiff={d:.3e}")
        except Exception as e:
            all_pass = False
            print(f"FAIL {t}  {e}")

    scalar_map = {
        "8.1": "testout8_1.csv",
        "8.2": "testout8_2.csv",
        "8.3": "testout8_3.csv",
    }

    for t in scalar_tests:
        exp = os.path.join(COURSE_DATA, scalar_map[t])
        got = os.path.join(OUT_DIR, scalar_map[t])
        try:
            d = check_scalar(exp, got, tol=TOL)
            diffs[t] = d
            print(f"PASS {t}  absdiff={d:.3e}")
        except Exception as e:
            all_pass = False
            print(f"FAIL {t}  {e}")

    if all_pass:
        print("PASS ✅ All outputs match expected.")
    else:
        print("FAIL ❌ We have mismatches.")

    if diffs:
        summary = ", ".join([f"{k}={v:.3e}" for k, v in diffs.items()])
        print("Diff summary:", summary)
        print(f"tol={TOL:.1e}")


if __name__ == "__main__":
    main()
