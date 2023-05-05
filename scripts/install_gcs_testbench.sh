#!/usr/bin/env bash
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

set -e

if [ "$(uname -m)" != "x86_64" ]; then
  echo "GCS testbench won't install on non-x86 architecture"
  exit 0
fi

if [ "$#" -ne 1 ]; then
  if [[ $(${PYTHON:-python3} --version | cut -d' ' -f2 | cut -d'.' -f2) -gt 6 ]]; then
    version="v0.35.0"
  else
    version="v0.18.0"
  fi
else
  version=$1
fi

${PYTHON:-python3} -m pip install \
   https://github.com/googleapis/storage-testbench/archive/refs/tags/${version}.tar.gz
