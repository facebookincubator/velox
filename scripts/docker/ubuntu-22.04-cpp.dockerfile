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
ARG base=ubuntu:22.04
FROM ${base}

RUN apt update && \
      apt install -y sudo \
            lsb-release \
            pip \
            python3


COPY scripts /velox/scripts/
COPY CMake/resolve_dependency_modules/arrow/cmake-compatibility.patch /
COPY CMake/resolve_dependency_modules/arrow/arrow-testing-boost.patch /

ENV VELOX_ARROW_CMAKE_PATCH="/cmake-compatibility.patch /arrow-testing-boost.patch" \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_INSTALL_DIR=/usr/local/bin

# TZ and DEBIAN_FRONTEND="noninteractive"
# are required to avoid tzdata installation
# to prompt for region selection.
ARG DEBIAN_FRONTEND="noninteractive"
# Set a default timezone, can be overriden via ARG
ARG tz="Etc/UTC"
ENV TZ=${tz}
RUN /bin/bash -o pipefail /velox/scripts/setup-ubuntu.sh

# Install tools needed for CI (gh for GitHub Actions stash, jq for JSON parsing)
RUN apt-get update && \
      apt-get install -y -q --no-install-recommends jq && \
      curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
      apt-get update && apt-get install -y -q --no-install-recommends gh && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

# Pre-download gflags source for BUNDLED builds to avoid downloading at build time.
RUN mkdir -p /velox/deps-sources && \
    curl -fsSL -o /velox/deps-sources/gflags-v2.2.2.tar.gz \
      https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz

WORKDIR /velox
