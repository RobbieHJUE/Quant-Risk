import os

COURSE_DATA = r"course/testfiles/data"
OUT_DIR = r"HW02/out"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("HW02 run_all.py scaffold ready.")
    print("COURSE_DATA =", COURSE_DATA)
    print("OUT_DIR     =", OUT_DIR)
    # TODO: implement tasks 1.x, 2.x, 3.x, 4.1, 6.x and write outputs to OUT_DIR

if __name__ == "__main__":
    main()
