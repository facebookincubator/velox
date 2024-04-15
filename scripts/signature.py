# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
import sys
from typing import Any

from deepdiff import DeepDiff

import pyvelox.pyvelox as pv

# Utility to export and diff function signatures.


# From https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"


aggregate_pattern = re.compile("(.*)(_merge|_merge_extract|_partial)")


def get_error_string(error_message):
    return f"""
Incompatible changes in function signatures have been detected.

{error_message}

Changing or removing function signatures breaks backwards compatibility as some users may rely on function signatures that no longer exist.

"""


def set_gh_output(name: str, value: Any):
    """Sets a Github Actions output variable. Only single line values are supported.
    value will be converted to a lower case string."""
    value = str(value).lower()

    if "\n" in value:
        raise ValueError("Only single line values are supported.")

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def show_error(error_message, error_path):
    if error_path:
        with open(error_path, "a+") as f:
            f.writelines(get_error_string(error_message))

    print(get_error_string(error_message))


def export(args):
    """Exports Velox function signatures."""
    pv.clear_signatures()

    if args.spark:
        pv.register_spark_signatures()

    if args.presto:
        pv.register_presto_signatures()

    signatures = pv.get_function_signatures()

    # Convert signatures to json
    jsoned_signatures = {}
    for key in signatures.keys():
        jsoned_signatures[key] = [str(value) for value in signatures[key]]

    # Persist to file
    with open(args.output_file, "w") as f:
        json.dump(jsoned_signatures, f)

    return 0


def export_aggregates(args):
    """Exports Velox Aggregate function signatures."""
    pv.clear_aggregate_signatures()

    if args.spark:
        pv.register_spark_aggregate_signatures()

    if args.presto:
        pv.register_presto_aggregate_signatures()

    signatures = pv.get_aggregate_function_signatures()

    # Convert signatures to json
    jsoned_signatures = {}
    for key in signatures.keys():
        jsoned_signatures[key] = [str(value) for value in signatures[key]]

    # Persist to file
    json.dump(jsoned_signatures, args.output_file)
    return 0


def diff_signatures(base_signatures, contender_signatures, error_path=""):
    """Diffs Velox function signatures. Returns a tuple of the delta diff and exit status"""

    delta = DeepDiff(
        base_signatures,
        contender_signatures,
        ignore_order=True,
        cutoff_distance_for_pairs=0.9,
        report_repetition=True,
        view="tree",
    )
    exit_status = 0
    if delta:
        if "dictionary_item_removed" in delta:
            error_message = ""
            for dic_removed in delta["dictionary_item_removed"]:
                error_message += (
                    f"""Function '{dic_removed.get_root_key()}' has been removed.\n"""
                )
            show_error(error_message, error_path)
            exit_status = 1

        if "values_changed" in delta:
            error_message = ""
            for value_change in delta["values_changed"]:
                error_message += f"""'{value_change.get_root_key()}{value_change.t1}' is changed to '{value_change.get_root_key()}{value_change.t2}'.\n"""
            show_error(error_message, error_path)
            exit_status = 1

        if "repetition_change" in delta:
            error_message = ""
            for rep_change in delta["repetition_change"]:
                error_message += f"""'{rep_change.get_root_key()}{rep_change.t1}' is repeated {rep_change.repetition['new_repeat']} times.\n"""
            show_error(error_message, error_path)
            exit_status = 1

        if "iterable_item_removed" in delta:
            error_message = ""
            for iter_change in delta["iterable_item_removed"]:
                error_message += f"""{iter_change.get_root_key()} has its function signature '{iter_change.t1}' removed.\n"""
            show_error(error_message, error_path)
            exit_status = 1

    else:
        print(f"{bcolors.BOLD}No differences found.")

    return delta, exit_status


def diff(args):
    """Diffs Velox function signatures."""
    with open(args.base) as f:
        base_signatures = json.load(f)

    with open(args.contender) as f:
        contender_signatures = json.load(f)
    return diff_signatures(base_signatures, contender_signatures)[1]


def bias(args):
    with open(args.base) as f:
        base_signatures = json.load(f)

    with open(args.contender) as f:
        contender_signatures = json.load(f)

    tickets = args.ticket_value
    bias_output, status = bias_signatures(
        base_signatures, contender_signatures, tickets, args.error_path
    )

    if bias_output:
        with open(args.output_path, "w") as f:
            print(f"{bias_output}", file=f, end="")

    return status


def bias_signatures(base_signatures, contender_signatures, tickets, error_path):
    """Returns newly added functions as string and a status flag.
    Newly added functions are biased like so `fn_name1=<ticket_count>,fn_name2=<ticket_count>`.
    If it detects incompatible changes returns 1 in the status.
    """
    delta, status = diff_signatures(base_signatures, contender_signatures, error_path)

    if not delta:
        print(f"{bcolors.BOLD} No changes detected: Nothing to do!")
        return "", status

    function_set = set()
    for items in delta.values():
        for item in items:
            function_set.add(item.get_root_key())

    if function_set:
        return f"{f'={tickets},'.join(sorted(function_set)) + f'={tickets}'}", status

    return "", status


def bias_aggregates(args):
    """
    Finds and exports aggregates whose signatures have been modified agasint a baseline.
    Saves the results to a file and sets a Github Actions Output.
    Currently this is hardcoded to presto aggregates.
    """
    with open(args.base) as f:
        base_signatures = json.load(f)

    with open(args.contender) as f:
        contender_signatures = json.load(f)

    delta, status = diff_signatures(
        base_signatures, contender_signatures, args.error_path
    )

    set_gh_output("presto_aggregate_error", status == 1)

    if not delta:
        print(f"{bcolors.BOLD} No changes detected: Nothing to do!")
        return status

    function_set = set()
    for items in delta.values():
        for item in items:
            fn_name = item.get_root_key()
            pattern = aggregate_pattern.match(fn_name)
            if pattern:
                function_set.add(pattern.group(1))
            else:
                function_set.add(fn_name)

    if function_set:
        biased_functions = ",".join(function_set)
        with open(args.output_path, "w") as f:
            print(f"{biased_functions}", file=f, end="")

        set_gh_output("presto_aggregate_functions", True)

    return 0


def gh_bias_check(args):
    """
    Exports signatures for the given group(s) and checks them for changes compared to a baseline.
    Saves the results to a file and sets a Github Actions Output for each group.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        print("This command is meant to be run in a Github Actions environment.")
        return 1

    # export signatures for each group
    for group in args.group:
        print(f"Exporting {group} signatures...")
        export_args = parse_args(
            [
                "export",
                f"--{group}",
                os.path.join(args.signature_dir, group + args.contender_postfix),
            ]
        )
        export(export_args)

    # compare signatures for each group
    for group in args.group:
        print(f"Comparing {group} signatures...")
        bias_args = parse_args(
            [
                "bias",
                os.path.join(args.signature_dir, group + args.base_postfix),
                os.path.join(args.signature_dir, group + args.contender_postfix),
                os.path.join(args.signature_dir, group + args.output_postfix),
                os.path.join(args.signature_dir, group + "_errors"),
            ]
        )

        bias_status = bias(bias_args)
        set_gh_output(f"{group}_error", bias_status == 1)

        # check if there are any changes that require the bias fuzzer to run
        has_tickets = os.path.isfile(
            os.path.join(args.signature_dir, group + args.output_postfix)
        )
        set_gh_output(f"{group}_functions", has_tickets)


def get_tickets(val):
    tickets = int(val)
    if tickets < 0:
        raise argparse.ArgumentTypeError("Cant have negative values!")
    return tickets


def parse_args(args):
    global parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Velox Function Signature Utility""",
    )

    command = parser.add_subparsers(dest="command")
    export_command_parser = command.add_parser("export")
    export_command_parser.add_argument("--spark", action="store_true")
    export_command_parser.add_argument("--presto", action="store_true")
    export_command_parser.add_argument("output_file", type=str)

    export_aggregates_command_parser = command.add_parser("export_aggregates")
    export_aggregates_command_parser.add_argument("--spark", action="store_true")
    export_aggregates_command_parser.add_argument("--presto", action="store_true")
    export_aggregates_command_parser.add_argument(
        "output_file", type=argparse.FileType("w")
    )

    diff_command_parser = command.add_parser("diff")
    diff_command_parser.add_argument("base", type=str)
    diff_command_parser.add_argument("contender", type=str)

    bias_command_parser = command.add_parser("bias")
    bias_command_parser.add_argument("base", type=str)
    bias_command_parser.add_argument("contender", type=str)
    bias_command_parser.add_argument("output_path", type=str)
    bias_command_parser.add_argument(
        "ticket_value", type=get_tickets, default=10, nargs="?"
    )
    bias_command_parser.add_argument("error_path", type=str, default="")

    gh_command_parser = command.add_parser("gh_bias_check")
    gh_command_parser.add_argument(
        "group",
        nargs="+",
        help='One or more group names to check for changed signatures. e.g. "spark" or "presto"',
        type=str,
    )
    gh_command_parser.add_argument(
        "--signature_dir", type=str, default="/tmp/signatures"
    )
    gh_command_parser.add_argument(
        "--base_postfix", type=str, default="_signatures_main.json"
    )
    gh_command_parser.add_argument(
        "--contender_postfix", type=str, default="_signatures_contender.json"
    )
    gh_command_parser.add_argument(
        "--output_postfix", type=str, default="_bias_functions"
    )

    bias_aggregate_command_parser = command.add_parser("bias_aggregates")
    bias_aggregate_command_parser.add_argument("base", type=str)
    bias_aggregate_command_parser.add_argument("contender", type=str)
    bias_aggregate_command_parser.add_argument("output_path", type=str)
    bias_aggregate_command_parser.add_argument("error_path", type=str, default="")

    parser.set_defaults(command="help")

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
