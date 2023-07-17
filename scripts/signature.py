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
import sys
import os.path
import random
import subprocess
import sys


import pyvelox.pyvelox as pv
from deepdiff import DeepDiff


# Utility to export and diff function signatures.


# From https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"


def export(args):
    """Exports Velox function signatures."""
    if args.spark:
        pv.register_spark_signatures("spark_")

    if args.presto:
        pv.register_presto_signatures("presto_")

    signatures = pv.get_function_signatures()

    # Convert signatures to json
    jsoned_signatures = {}
    for key in signatures.keys():
        jsoned_signatures[key] = [str(value) for value in signatures[key]]

    # Persist to file
    json.dump(jsoned_signatures, args.output_file)
    return 0


def diff_signatures(args):
    """Diffs Velox function signatures. Returns a tuple of the delta diff and exit status"""
    first_signatures = json.load(args.first)
    second_signatures = json.load(args.second)
    delta = DeepDiff(
        first_signatures,
        second_signatures,
        ignore_order=True,
        report_repetition=True,
        view="tree",
    )
    exit_status = 0
    if delta:
        if "dictionary_item_removed" in delta:
            print(
                f"Signature removed: {bcolors.FAIL}{delta['dictionary_item_removed']}"
            )
            exit_status = 1

        if "values_changed" in delta:
            print(f"Signature changed: {bcolors.FAIL}{delta['values_changed']}")
            exit_status = 1

        if "repetition_change" in delta:
            print(f"Signature repeated: {bcolors.FAIL}{delta['repetition_change']}")
            exit_status = 1

        if "iterable_item_removed" in delta:
            print(
                f"Iterable item removed: {bcolors.FAIL}{delta['iterable_item_removed']}"
            )
            exit_status = 1

        print(f"Found differences: {bcolors.OKGREEN}{delta}")

    else:
        print(f"{bcolors.BOLD}No differences found.")

    if exit_status:
        print(
            f""" 
            {bcolors.BOLD}Incompatible changes in function signatures have been detected.
            This means your changes have modified function signatures and possibly broken backwards compatibility.  
        """
        )

    return delta, exit_status


def diff(args):
    """Diffs Velox function signatures."""
    return diff_signatures(args)[1]


def bias(args):
    """Biases a provided fuzzer with newly added functions."""

    delta, status = diff_signatures(args)

    # Return if the signature check call flags incompatible changes.
    if status:
        return status

    if not len(delta):
        print(f"{bcolors.BOLD} No changes detected: Nothing to do!")
        return 0

    function_set = set()
    for items in delta.values():
        for item in items:
            function_set.add(item.get_root_key())

    # Split functions by presto, spark.
    function_dict = {}
    for item in function_set:
        split = item.split("_")
        function_dict.setdefault(split[0], []).append(split[1])

    print(f"{bcolors.BOLD}Functions to be biased: {function_dict}")

    # Create the executable string
    fuzzer_path = args.presto_fuzzer_path
    fuzzer_output = "/tmp/fuzzer.log"
    # Currently only support Presto.
    command = [
        fuzzer_path,
        f"--seed {random.randint(0, 99999)}",
        f"--assign_function_tickets {'=10,'.join(function_dict['presto']) + '=10'}",
        "--lazy_vector_generation_ratio 0.2",
        "--duration_sec 3600 --enable_variadic_signatures",
        "--velox_fuzzer_enable_complex_types",
        "--velox_fuzzer_enable_column_reuse",
        "--velox_fuzzer_enable_expression_reuse",
        "--max_expression_trees_per_step 2",
        "--retry_with_try",
        "--enable_dereference",
        "--logtostderr=1 --minloglevel=0",
        "--repro_persist_path=/tmp/fuzzer_repro",
    ]

    print(f"Going to run command: {command}")

    with open(fuzzer_output, "wb") as f:
        process = subprocess.Popen(
            command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True
        )
        # replace "" with b"" for Python 3
        for line in iter(process.stdout.readline, b""):
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)

        return process.returncode


def check_fuzzer_executable(parser, arg):
    if not os.path.exists(arg):
        parser.error(f"The provided fuzzer path: {arg} doesnt exist!")
    else:
        return arg


def parse_args():
    global parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Velox Function Signature Utility""",
    )

    command = parser.add_subparsers(dest="command")
    export_command_parser = command.add_parser("export")
    export_command_parser.add_argument("--spark", action="store_true")
    export_command_parser.add_argument("--presto", action="store_false")
    export_command_parser.add_argument("output_file", type=argparse.FileType("w"))

    diff_command_parser = command.add_parser("diff")
    diff_command_parser.add_argument("first", type=argparse.FileType("r"))
    diff_command_parser.add_argument("second", type=argparse.FileType("r"))

    bias_command_parser = command.add_parser("bias")
    bias_command_parser.add_argument("first", type=argparse.FileType("r"))
    bias_command_parser.add_argument("second", type=argparse.FileType("r"))
    bias_command_parser.add_argument(
        "presto_fuzzer_path",
        type=lambda arg: check_fuzzer_executable(bias_command_parser, arg),
    )

    parser.set_defaults(command="help")

    return parser.parse_args()


def main():
    args = parse_args()
    return globals()[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
