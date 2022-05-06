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

SCRIPTDIR=$(dirname "$0")
source $SCRIPTDIR/setup-helper-functions.sh

# Propagate errors and improve debugging.
set -eufx -o pipefail

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh

function install_gcs-sdk-cpp {
  #install google-cloud-cpp dependencies (ubuntu 20.04)
  echo " Installing apt dependencies..."
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update && \
  sudo apt-get --no-install-recommends install -y apt-transport-https apt-utils \
          automake build-essential ccache cmake ca-certificates curl git \
          gcc g++ libc-ares-dev libc-ares2 libcurl4-openssl-dev libre2-dev \
          libssl-dev m4 make pkg-config tar wget zlib1g-dev

  #install abseil
  echo " Installing abseil..."
  mkdir -p $HOME/Downloads/abseil-cpp && cd $HOME/Downloads/abseil-cpp
  curl -sSL https://github.com/abseil/abseil-cpp/archive/20211102.0.tar.gz | \
      tar -xzf - --strip-components=1 && \
      sed -i 's/^#define ABSL_OPTION_USE_\(.*\) 2/#define ABSL_OPTION_USE_\1 0/' "absl/base/options.h" && \
      cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=yes \
        -DCMAKE_CXX_STANDARD=11 \
        -H. -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  sudo cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  sudo ldconfig


  #install protobuf
  echo " Installing protobuf..."
  mkdir -p $HOME/Downloads/protobuf && cd $HOME/Downloads/protobuf
  curl -sSL https://github.com/protocolbuffers/protobuf/archive/v3.20.1.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=yes \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Hcmake -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  sudo cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  sudo ldconfig

  #install gRPC
  echo " Installing gRPC..."
  mkdir -p $HOME/Downloads/grpc && cd $HOME/Downloads/grpc
  curl -sSL https://github.com/grpc/grpc/archive/v1.45.2.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_TESTS=OFF \
          -DgRPC_ABSL_PROVIDER=package \
          -DgRPC_CARES_PROVIDER=package \
          -DgRPC_PROTOBUF_PROVIDER=package \
          -DgRPC_RE2_PROVIDER=package \
          -DgRPC_SSL_PROVIDER=package \
          -DgRPC_ZLIB_PROVIDER=package \
          -H. -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  sudo cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  sudo ldconfig

  #install crc32c
  echo " Installing crc32c..."
  mkdir -p $HOME/Downloads/crc32c && cd $HOME/Downloads/crc32c
  curl -sSL https://github.com/google/crc32c/archive/1.1.2.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=yes \
          -DCRC32C_BUILD_TESTS=OFF \
          -DCRC32C_BUILD_BENCHMARKS=OFF \
          -DCRC32C_USE_GLOG=OFF \
          -H. -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  sudo cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  sudo ldconfig

  #install nlohmann_json library
  echo " Installing nlohmann json..."
  mkdir -p $HOME/Downloads/json && cd $HOME/Downloads/json
  curl -sSL https://github.com/nlohmann/json/archive/v3.10.5.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=yes \
        -DBUILD_TESTING=OFF \
        -DJSON_BuildTests=OFF \
        -H. -Bcmake-out/nlohmann/json && \
  sudo cmake --build cmake-out/nlohmann/json --target install -- -j ${NCPU:-4} && \
  sudo ldconfig

  echo " Clone, compile and install google-cloud-cpp..."
  #clone and compile the main project
  git clone https://github.com/googleapis/google-cloud-cpp.git ${DEPENDENCY_DIR}/google-cloud-cpp
  cd ${DEPENDENCY_DIR}/google-cloud-cpp
  # Pick a location to install the artifacts, e.g., `/usr/local` or `/opt`
  PREFIX="${DEPENDENCY_DIR}/google-cloud-cpp-installed"
  cmake -H. -Bcmake-out \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_MESSAGE=NEVER \
    -DBUILD_TESTING=OFF \
    -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF \
    -DGOOGLE_CLOUD_CPP_ENABLE=storage
  cmake --build cmake-out -- -j "$(nproc)"
  cmake --build cmake-out --target install

}

function install_aws-sdk-cpp {
  local AWS_REPO_NAME="aws/aws-sdk-cpp"
  local AWS_SDK_VERSION="1.9.96"

  github_checkout $AWS_REPO_NAME $AWS_SDK_VERSION --depth 1 --recurse-submodules
  cmake_install -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS:BOOL=OFF -DMINIMIZE_SIZE:BOOL=ON -DENABLE_TESTING:BOOL=OFF -DBUILD_ONLY:STRING="s3;identity-management" -DCMAKE_INSTALL_PREFIX="${DEPENDENCY_DIR}/install"
}

function install_libhdfs3 {
  github_checkout apache/hawq master
  cd $DEPENDENCY_DIR/hawq/depends/libhdfs3
  if [[ "$OSTYPE" == darwin* ]]; then
     sed -i '' -e "/FIND_PACKAGE(GoogleTest REQUIRED)/d" ./CMakeLists.txt
     sed -i '' -e "s/dumpversion/dumpfullversion/" ./CMakeLists.txt
  fi

  if [[ "$OSTYPE" == linux-gnu* ]]; then
    sed -i "/FIND_PACKAGE(GoogleTest REQUIRED)/d" ./CMakeLists.txt
    sed -i "s/dumpversion/dumpfullversion/" ./CMake/Platform.cmake
  fi
  cmake_install
}

DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}
cd "${DEPENDENCY_DIR}" || exit
# aws-sdk-cpp missing dependencies

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
   yum -y install libxml2-devel libgsasl-devel libuuid-devel
fi

if [[ "$OSTYPE" == darwin* ]]; then
   brew install libxml2 gsasl
fi


install_aws=0
install_gcs=0
install_hdfs=0

if [ "$#" -eq 0 ]; then
    install_aws=1 #install it if no other arguments are specified
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    gcs)
      install_gcs=1
      shift # past argument
      ;;
    aws)
      install_aws=1
      shift # past argument
      ;;
    hdfs)
      install_hdfs=1
      shift # past argument
      ;;
    *)
      echo "ERROR: Unknown option $1! will be ignored!"
      shift
      ;;
  esac
done

if [ $install_gcs -eq 1 ]; then
  install_gcs-sdk-cpp
fi

if [ $install_aws -eq 1 ]; then
  install_aws-sdk-cpp
fi
if [ $install_hdfs -eq 1 ]; then
  install_libhdfs3
fi

_ret=$?
if [ $_ret -eq 0 ] ; then
   echo "All deps for Velox adapters installed!"
fi
