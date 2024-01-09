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
# Build the test and build container for presto_cpp
#
FROM ghcr.io/facebookincubator/velox-dev:circleci-avx

RUN dnf install java-1.8.0-openjdk wget\
    && wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio-20220526054841.0.0.x86_64.rpm \
    && rpm -i minio-20220526054841.0.0.x86_64.rpm \
    && rm minio-20220526054841.0.0.x86_64.rpm \
