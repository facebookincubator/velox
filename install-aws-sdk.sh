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

if [ ! -d "aws-sdk-cpp" ]; then
    git clone https://github.com/aws/aws-sdk-cpp --recurse-submodules
fi
cd aws-sdk-cpp

mkdir -p build
cd build

cmake ../ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_ONLY="s3;sts;cognito-identity;identity-management" \
  -DENABLE_TESTING=OFF
cmake --build . --config=Debug
cmake --install . --config=Debug

#cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/local -DCMAKE_INSTALL_PREFIX=/usr/local
#make
#sudo make install
