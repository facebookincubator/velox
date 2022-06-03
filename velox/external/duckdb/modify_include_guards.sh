#!/bin/bash
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

# This script documents setting up a macOS host for presto_cpp
# development.  Running it should make you ready to compile.
#
# Environment variables:
# * INSTALL_PREREQUISITES="N": Skip installation of brew/pip deps.
# * PROMPT_ALWAYS_RESPOND="n": Automatically respond to interactive prompts.
#     Use "n" to never wipe directories.
#
# You can also run individual functions below by specifying them as arguments:
# $ scripts/setup-macos.sh install_googletest install_fmt
#

echo $1
TAGS=$(sed -n 's/^#ifndef \(_THRIFT_[A-Z][A-Z_]*_\)/\1/p' ${1})
PREFIX=_DUCKDB
# Substitute each tag for the same tag with '_DUCKDB' prepended.
for TAG in ${TAGS}
do
	echo "Substituting ${TAG} with ${PREFIX}${TAG} in ${1}"
	COMMAND="s/\(${TAG}\)/${PREFIX}\1/"
	echo $COMMAND
	$(sed -i '' ${COMMAND} ${1})
done
