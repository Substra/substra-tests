#!/usr/bin/env python3

import argparse

from ci import slack


def main():
    parser = argparse.ArgumentParser()

    class ParseOptInt(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                setattr(namespace, self.dest, int(values))
            except ValueError:
                setattr(namespace, self.dest, None)

    parser.add_argument("hook")
    parser.add_argument("desc_file", help="Path to a file with the contents of the message")
    parser.add_argument("--title", help="Report title")
    parser.add_argument("--status", help="Reported job status")
    parser.add_argument("--link", help="Link to job")
    parser.add_argument("--duration-seconds", help="The end-to-end tests duration in seconds", action=ParseOptInt)

    args = parser.parse_args()

    try:
        with open(args.desc_file) as fp:
            message = fp.read()
    except FileNotFoundError:
        message = "🔴 Something wrong happened"
    slack.send_message(args.hook, args.title, message, args.status, args.link, args.duration_seconds)


if __name__ == "__main__":
    main()
