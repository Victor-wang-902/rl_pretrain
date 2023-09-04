import argparse
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    args = parser.parse_args()
    print(sys.argv)