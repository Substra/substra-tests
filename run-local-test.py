import argparse
import subprocess
import sys
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-summary-to-file",
        type=str,
        default=None,
        help="Write a summary of the results to the given filename",
    )
    args = parser.parse_args()
    return args


def write_summary_file(test_passed, path: Path):
    with path.open("w") as fp:
        if test_passed is not None:
            res = "✅" if test_passed else "❌"
        else:
            res = "⏭ (skipped)"
        fp.write(f"{res} sdk local tests - python version {sys.version_info.major}.{sys.version_info.minor} \n")


def run_test_local():
    try:
        subprocess.check_call(["make test-local"], shell=True)
        return True
    except subprocess.CalledProcessError:
        print("FATAL: `make test-local` completed with a non-zero exit code. Did some test(s) fail?")
        return False


def main():
    args = arg_parse()

    test_passed = run_test_local()

    summary_file = args.write_summary_to_file
    if summary_file:
        write_summary_file(test_passed, Path(summary_file))

    sys.exit(0 if test_passed else 1)


if __name__ == "__main__":
    main()
