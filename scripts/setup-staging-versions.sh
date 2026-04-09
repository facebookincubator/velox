#!/bin/bash
# shellcheck disable=SC2034
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

# Staging version overrides for dependency upgrades.
# This file is sourced by setup-versions.sh when VELOX_STAGING=true,
# allowing staging Docker images to build with upcoming dependency
# versions while production images remain on current versions.
# See .github/README.md for the full staging workflow.
#
# When no upgrade is in progress, this file should be empty
# (no version overrides below this comment).

DUCKDB_VERSION="v1.4.4"
DUCKDB_GIT_COMMIT_HASH="6ddac80"
