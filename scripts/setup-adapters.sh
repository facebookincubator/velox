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
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}

function install_gcs-sdk-cpp {
  user=`whoami`
  case "$user" in
    root) echo 'Running as root' ; sudocmd="";;
       *) echo 'Running as other user than root' ; sudocmd="sudo";;
  esac
  mkdir -p ${DEPENDENCY_DIR}/gcs_artifacts/
  ARTIFACTS_PREFIX="${DEPENDENCY_DIR}/gcs_artifacts/"
  # install gcs dependencies
  # https://github.com/googleapis/google-cloud-cpp/blob/main/doc/packaging.md#required-libraries
  # install abseil
  echo " Installing abseil..."
  mkdir -p ${DEPENDENCY_DIR}/abseil-cpp && cd ${DEPENDENCY_DIR}/abseil-cpp
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
  $sudocmd cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  $sudocmd ldconfig


  # install protobuf
  echo " Installing protobuf..."
  mkdir -p ${DEPENDENCY_DIR}/protobuf && cd ${DEPENDENCY_DIR}/protobuf
  curl -sSL https://github.com/protocolbuffers/protobuf/archive/v3.20.1.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="${ARTIFACTS_PREFIX}" \
          -DCMAKE_MODULE_PATH="${ARTIFACTS_PREFIX}" \
          -DBUILD_SHARED_LIBS=yes \
          -Dprotobuf_BUILD_TESTS=OFF \
          -Hcmake -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  $sudocmd cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  $sudocmd ldconfig

  # install gRPC
  echo " Installing gRPC..."
  mkdir -p ${DEPENDENCY_DIR}/grpc && cd ${DEPENDENCY_DIR}/grpc
  curl -sSL https://github.com/grpc/grpc/archive/v1.45.2.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DgRPC_INSTALL=ON \
          -DCMAKE_INSTALL_PREFIX="${ARTIFACTS_PREFIX}" \
          -DgRPC_BUILD_TESTS=OFF \
          -DgRPC_ABSL_PROVIDER=package \
          -DCMAKE_MODULE_PATH="${ARTIFACTS_PREFIX}" \
          -DgRPC_CARES_PROVIDER=package \
          -DgRPC_PROTOBUF_PROVIDER=package \
          -DgRPC_RE2_PROVIDER=package \
          -DgRPC_SSL_PROVIDER=package \
          -DgRPC_ZLIB_PROVIDER=package \
          -H. -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  $sudocmd cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  $sudocmd ldconfig

  # install crc32c
  echo " Installing crc32c..."
  mkdir -p ${DEPENDENCY_DIR}/crc32c && cd ${DEPENDENCY_DIR}/crc32c
  curl -sSL https://github.com/google/crc32c/archive/1.1.2.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=yes \
          -DCMAKE_INSTALL_PREFIX="${ARTIFACTS_PREFIX}" \
          -DCRC32C_BUILD_TESTS=OFF \
          -DCMAKE_MODULE_PATH="${ARTIFACTS_PREFIX}" \
          -DCRC32C_BUILD_BENCHMARKS=OFF \
          -DCRC32C_USE_GLOG=OFF \
          -H. -Bcmake-out && \
      cmake --build cmake-out -- -j ${NCPU:-4} && \
  $sudocmd cmake --build cmake-out --target install -- -j ${NCPU:-4} && \
  $sudocmd ldconfig

  # install nlohmann_json library
  echo " Installing nlohmann json..."
  mkdir -p ${DEPENDENCY_DIR}/json && cd ${DEPENDENCY_DIR}/json
  curl -sSL https://github.com/nlohmann/json/archive/v3.10.5.tar.gz | \
      tar -xzf - --strip-components=1 && \
      cmake \
        -DCMAKE_INSTALL_PREFIX="${ARTIFACTS_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=yes \
        -DCMAKE_MODULE_PATH="${ARTIFACTS_PREFIX}" \
        -DBUILD_TESTING=OFF \
        -DJSON_BuildTests=OFF \
        -H. -Bcmake-out/nlohmann/json && \
  $sudocmd cmake --build cmake-out/nlohmann/json --target install -- -j ${NCPU:-4} && \
  $sudocmd ldconfig
  echo " Clone, compile and install google-cloud-cpp..."
  $sudocmd rm -rf ${DEPENDENCY_DIR}/google-cloud-cpp
  # clone and compile the main project
  git clone https://github.com/googleapis/google-cloud-cpp.git ${DEPENDENCY_DIR}/google-cloud-cpp
  cd ${DEPENDENCY_DIR}/google-cloud-cpp
  # Pick a location to install the artifacts, e.g., `/usr/local` or `/opt`
  PREFIX="/usr/local"
  cmake -H. -Bcmake-out \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_MESSAGE=NEVER \
    -DCMAKE_MODULE_PATH="${ARTIFACTS_PREFIX}" \
    -DBUILD_TESTING=OFF \
    -DCrc32c_DIR="${ARTIFACTS_PREFIX}/lib/cmake/Crc32c/" \
    -Dnlohmann_json_DIR="${ARTIFACTS_PREFIX}/lib/cmake/nlohmann_json" \
    -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF \
    -DGOOGLE_CLOUD_CPP_ENABLE=storage
  $sudocmd cmake --build cmake-out -- -j "$(nproc)"
  $sudocmd cmake --build cmake-out --target install
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
    #install all if none is specifically picked
    install_aws=1
    install_gcs=1
    install_hdfs=1
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
  # aws-sdk-cpp missing dependencies
  yum install -y curl-devel

  install_aws-sdk-cpp
fi
if [ $install_hdfs -eq 1 ]; then
  install_libhdfs3
fi

_ret=$?
if [ $_ret -eq 0 ] ; then
   echo "All deps for Velox adapters installed!"
fi
