import numpy as np
import pandas as pd

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW01/out"

TOL = 1e-5  

def check(expected, got):
    a = pd.read_csv(expected).to_numpy(float)
    b = pd.read_csv(got).to_numpy(float)
    d = np.max(np.abs(a - b))
    return d

def main():
    d1 = check(f"{COURSE_DATA}/testout7_1.csv", f"{OUT_DIR}/testout_7.1.csv")
    d2 = check(f"{COURSE_DATA}/testout7_2.csv", f"{OUT_DIR}/testout_7.2.csv")
    d3 = check(f"{COURSE_DATA}/testout7_3.csv", f"{OUT_DIR}/testout_7.3.csv")

    print("Max diffs:", d1, d2, d3)
    if max(d1, d2, d3) <= TOL:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
