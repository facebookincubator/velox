# Copyright (c) Meta Platforms, Inc. and its affiliates.
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

"""Acts as a Python interpreter with pyarrow for the C++ tests: runs the script
given as the first argument as __main__, with the rest as its arguments."""

import runpy
import sys

# The worker script imports pyarrow, so the interpreter bundles it.
# @dep=fbsource//third-party/pypi/pyarrow:pyarrow


def main() -> None:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
