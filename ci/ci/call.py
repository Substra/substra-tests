import subprocess
from typing import List
from typing import TextIO


def call(
    cmd: str,
    print_cmd: bool = True,
    secrets: List[str] = None,
    stdout: TextIO = None,
) -> int:
    if not secrets:
        secrets = []

    if print_cmd:
        printed_cmd = f"+ {cmd}"

        for secret in secrets:
            printed_cmd = printed_cmd.replace(secret, "****")

        print(printed_cmd)
    return subprocess.check_call([cmd], shell=True, stdout=stdout, stderr=stdout)


def call_output(cmd: str, print_cmd: bool = True, no_stderr: bool = False) -> str:
    if print_cmd:
        print(f"+ {cmd}")

    if no_stderr:
        stderr = subprocess.DEVNULL
    else:
        stderr = subprocess.STDOUT

    res = subprocess.check_output([cmd], shell=True, stderr=stderr)
    return res.decode().strip()
