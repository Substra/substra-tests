#!/usr/bin/env python3

import argparse

from ci import slack


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("hook")
    parser.add_argument("desc_file", help="Path to a file with the contents of the message")
    parser.add_argument("--title", default="Connect e2e tests")
    parser.add_argument("--context")

    args = parser.parse_args()

    try:
        with open(args.desc_file) as fp:
            message = fp.read()
    except FileNotFoundError:
        message = "🔴 Something wrong happened"
    slack.send_message(args.hook, args.title, message, args.context)


if __name__ == "__main__":
    main()
