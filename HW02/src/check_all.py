import os
import numpy as np
import pandas as pd

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW02/out"

TOL = 1e-6

def _read_numeric_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    # If first column is non-numeric labels, drop it
    if df.shape[1] >= 2 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.iloc[:, 1:]

    return df.to_numpy(dtype=float)


def check_one(expected_path: str, got_path: str, tol: float = TOL) -> float:
    if not os.path.exists(expected_path):
        raise FileNotFoundError(f"Expected file not found: {expected_path}")
    if not os.path.exists(got_path):
        raise FileNotFoundError(f"Your output file not found: {got_path}")

    a = _read_numeric_matrix(expected_path)
    b = _read_numeric_matrix(got_path)

    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: expected {a.shape}, got {b.shape}")

    diff = np.abs(a - b)
    maxdiff = float(np.max(diff)) if diff.size else 0.0

    if maxdiff > tol:
        i, j = np.argwhere(diff > tol)[0]
        raise AssertionError(
            f"Value mismatch (maxdiff={maxdiff:.3e}) at [{i},{j}]: expected {a[i,j]}, got {b[i,j]}"
        )

    return maxdiff


def get_paths_for_test(t: str) -> tuple[str, str]:
    # Special naming cases
    name_map = {
        # Part 6 uses testout6_1 / testout6_2 naming (no dot)
        "6.1": ("testout6_1.csv", "testout6_1.csv"),
        "6.2": ("testout6_2.csv", "testout6_2.csv"),
    }

    if t in name_map:
        exp_name, got_name = name_map[t]
    else:
        exp_name = f"testout_{t}.csv"
        got_name = f"testout_{t}.csv"

    exp = os.path.join(COURSE_DATA, exp_name)
    got = os.path.join(OUT_DIR, got_name)
    return exp, got


def main():
    tests = [
        "1.1", "1.2", "1.3", "1.4",
        "2.1", "2.2", "2.3",
        "3.1", "3.2", "3.3", "3.4",
        "4.1",
        "6.1", "6.2",
    ]

    diffs = {}
    all_pass = True

    for t in tests:
        exp, got = get_paths_for_test(t)
        try:
            d = check_one(exp, got, tol=TOL)
            diffs[t] = d
            print(f"PASS {t}  maxdiff={d:.3e}")
        except Exception as e:
            all_pass = False
            print(f"FAIL {t}  {e}")

    if all_pass:
        print("PASS ✅ ALL part outputs match expected.")
    else:
        print("FAIL ❌ We have mismatches.")

    if diffs:
        md = max(diffs.values())
        print("Max diffs:", ", ".join([f"{k}={v:.3e}" for k, v in diffs.items()]))
        print(f"Overall maxdiff={md:.3e}  (tol={TOL:.1e})")


if __name__ == "__main__":
    main()
