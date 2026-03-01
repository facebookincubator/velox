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

set -e
set -o pipefail
set -u

# APT update.
apt-get update

# Install essentials.
apt-get install -y sudo locales wget tar tzdata git ccache ninja-build build-essential \
  llvm-14-dev clang-14 libiberty-dev libdwarf-dev libre2-dev libz-dev \
  liblzo2-dev libzstd-dev libsnappy-dev libdouble-conversion-dev libssl-dev \
  libboost-all-dev libcurl4-openssl-dev curl zip unzip tar pkg-config \
  autoconf-archive bison flex libfl-dev libc-ares-dev libicu-dev \
  libgoogle-glog-dev libbz2-dev libgflags-dev libgmock-dev libevent-dev \
  liblz4-dev libsodium-dev libelf-dev \
  autoconf automake g++ libnuma-dev libtool numactl unzip libdaxctl-dev \
  openjdk-11-jdk \
  maven cmake \
  chrpath patchelf

# Install GCC 11.
apt-get install -y software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get install -y gcc-11 g++-11
rm -f /usr/bin/gcc /usr/bin/g++
ln -s /usr/bin/gcc-11 /usr/bin/gcc
ln -s /usr/bin/g++-11 /usr/bin/g++
cc --version
c++ --version
