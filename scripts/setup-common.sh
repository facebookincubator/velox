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

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh
source $SCRIPTDIR/setup-versions.sh

NPROC=$(getconf _NPROCESSORS_ONLN)
CMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}

BUILD_DUCKDB="${BUILD_DUCKDB:-true}"

USE_CLANG="${USE_CLANG:-false}"

MACHINE=$(uname -m)

function install_fmt {
  wget_and_untar https://github.com/fmtlib/fmt/archive/${FMT_VERSION}.tar.gz fmt
  cmake_install fmt -DFMT_TEST=OFF
}

function install_folly {
  wget_and_untar https://github.com/facebook/folly/archive/refs/tags/${FB_OS_VERSION}.tar.gz folly
  cmake_install folly -DBUILD_TESTS=OFF -DFOLLY_HAVE_INT128_T=ON
}

function install_fizz {
  wget_and_untar https://github.com/facebookincubator/fizz/archive/refs/tags/${FB_OS_VERSION}.tar.gz fizz
  cmake_install fizz/fizz -DBUILD_TESTS=OFF
}

function install_wangle {
  wget_and_untar https://github.com/facebook/wangle/archive/refs/tags/${FB_OS_VERSION}.tar.gz wangle
  cmake_install wangle/wangle -DBUILD_TESTS=OFF
}

function install_mvfst {
  wget_and_untar https://github.com/facebook/mvfst/archive/refs/tags/${FB_OS_VERSION}.tar.gz mvfst
  cmake_install mvfst -DBUILD_TESTS=OFF
}

function install_fbthrift {
  wget_and_untar https://github.com/facebook/fbthrift/archive/refs/tags/${FB_OS_VERSION}.tar.gz fbthrift
  cmake_install fbthrift -Denable_tests=OFF -DBUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
}

function install_duckdb {
  if $BUILD_DUCKDB ; then
    echo 'Building DuckDB'
    wget_and_untar https://github.com/duckdb/duckdb/archive/refs/tags/${DUCKDB_VERSION}.tar.gz duckdb
    cmake_install duckdb -DBUILD_UNITTESTS=OFF -DENABLE_SANITIZER=OFF -DENABLE_UBSAN=OFF -DBUILD_SHELL=OFF -DEXPORT_DLL_SYMBOLS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  fi
}

function install_boost {
  wget_and_untar https://github.com/boostorg/boost/releases/download/${BOOST_VERSION}/${BOOST_VERSION}.tar.gz boost
  (
    cd boost
    if [[ ${USE_CLANG} != "false" ]]; then
      ./bootstrap.sh --prefix=/usr/local --with-toolset="clang-15"
      # Switch the compiler from the clang-15 toolset which doesn't exist (clang-15.jam) to
      # clang of version 15 when toolset clang-15 is used.
      # This reconciles the project-config.jam generation with what the b2 build system allows for customization.
      sed -i 's/using clang-15/using clang : 15/g' project-config.jam
      ${SUDO} ./b2 "-j${NPROC}" -d0 install threading=multi toolset=clang-15 --without-python
    else
      ./bootstrap.sh --prefix=/usr/local
      ${SUDO} ./b2 "-j${NPROC}" -d0 install threading=multi --without-python
    fi
  )
}

function install_protobuf {
  wget_and_untar https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz protobuf
  cmake_install protobuf -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package
}

function install_double_conversion {
  wget_and_untar https://github.com/google/double-conversion/archive/refs/tags/${DOUBLE_CONVERSION_VERSION}.tar.gz double-conversion
  cmake_install double-conversion -DBUILD_TESTING=OFF
}

function install_ranges_v3 {
  wget_and_untar https://github.com/ericniebler/range-v3/archive/refs/tags/${RANGE_V3_VERSION}.tar.gz ranges_v3
  cmake_install ranges_v3 -DRANGES_ENABLE_WERROR=OFF -DRANGE_V3_TESTS=OFF -DRANGE_V3_EXAMPLES=OFF
}

function install_re2 {
  wget_and_untar https://github.com/google/re2/archive/refs/tags/${RE2_VERSION}.tar.gz re2
  cmake_install re2 -DRE2_BUILD_TESTING=OFF
}

function install_glog {
  wget_and_untar https://github.com/google/glog/archive/${GLOG_VERSION}.tar.gz glog
  cmake_install glog -DBUILD_SHARED_LIBS=ON
}

function install_lzo {
  wget_and_untar http://www.oberhumer.com/opensource/lzo/download/lzo-${LZO_VERSION}.tar.gz lzo
  (
    cd lzo
    ./configure --prefix=/usr --enable-shared --disable-static --docdir=/usr/share/doc/lzo-${LZO_VERSION}
    make "-j${NPROC}"
    ${SUDO} make install
  )
}

function install_snappy {
  wget_and_untar https://github.com/google/snappy/archive/${SNAPPY_VERSION}.tar.gz snappy
  cmake_install snappy -DSNAPPY_BUILD_TESTS=OFF
}

function install_xsimd {
  wget_and_untar https://github.com/xtensor-stack/xsimd/archive/refs/tags/${XSIMD_VERSION}.tar.gz xsimd
  cmake_install xsimd
}

function install_simdjson {
  wget_and_untar https://github.com/simdjson/simdjson/archive/refs/tags/v${SIMDJSON_VERSION}.tar.gz simdjson
  cmake_install simdjson
}

function install_arrow {
  wget_and_untar https://archive.apache.org/dist/arrow/arrow-${ARROW_VERSION}/apache-arrow-${ARROW_VERSION}.tar.gz arrow
  (
    cd arrow/cpp
    cmake_install \
      -DARROW_PARQUET=OFF \
      -DARROW_WITH_THRIFT=ON \
      -DARROW_WITH_LZ4=ON \
      -DARROW_WITH_SNAPPY=ON \
      -DARROW_WITH_ZLIB=ON \
      -DARROW_WITH_ZSTD=ON \
      -DARROW_JEMALLOC=OFF \
      -DARROW_SIMD_LEVEL=NONE \
      -DARROW_RUNTIME_SIMD_LEVEL=NONE \
      -DARROW_WITH_UTF8PROC=OFF \
      -DARROW_TESTING=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
      -DARROW_BUILD_STATIC=ON \
      -DThrift_SOURCE=BUNDLED

    # Install thrift.
    cd _build/thrift_ep-prefix/src/thrift_ep-build
    ${SUDO} cmake --install ./ --prefix /usr/local/
  )
}

# Adapters that can be installed.

function install_aws_deps {
  local AWS_REPO_NAME="aws/aws-sdk-cpp"

  github_checkout $AWS_REPO_NAME $AWS_SDK_VERSION --depth 1 --recurse-submodules
  cmake_install -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS:BOOL=OFF -DMINIMIZE_SIZE:BOOL=ON -DENABLE_TESTING:BOOL=OFF -DBUILD_ONLY:STRING="s3;identity-management"
}

function install_minio {
  local MINIO_OS=${1:-darwin}
  local MINIO_ARCH

  if [[ $MACHINE == aarch64 ]]; then
    MINIO_ARCH="arm64"
  elif [[ $MACHINE == x86_64 ]]; then
    MINIO_ARCH="amd64"
  else
    echo "Unsupported Minio platform"
  fi

  wget https://dl.min.io/server/minio/release/${MINIO_OS}-${MINIO_ARCH}/archive/minio.RELEASE.${MINIO_VERSION} -O ${MINIO_BINARY_NAME}
  chmod +x ./${MINIO_BINARY_NAME}
  ${SUDO} mv ./${MINIO_BINARY_NAME} /usr/local/bin/
}

function install_gcs-sdk-cpp {
  # Install gcs dependencies
  # https://github.com/googleapis/google-cloud-cpp/blob/main/doc/packaging.md#required-libraries

  # abseil-cpp
  github_checkout abseil/abseil-cpp ${ABSEIL_VERSION} --depth 1
  cmake_install \
    -DABSL_BUILD_TESTING=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DABSL_ENABLE_INSTALL=ON

  # protobuf
  github_checkout protocolbuffers/protobuf v${PROTOBUF_VERSION} --depth 1
  cmake_install \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_ABSL_PROVIDER=package

  # grpc
  github_checkout grpc/grpc ${GRPC_VERSION} --depth 1
  cmake_install \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_ABSL_PROVIDER=package \
    -DgRPC_ZLIB_PROVIDER=package \
    -DgRPC_CARES_PROVIDER=package \
    -DgRPC_RE2_PROVIDER=package \
    -DgRPC_SSL_PROVIDER=package \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DgRPC_INSTALL=ON

  # crc32
  github_checkout google/crc32c ${CRC32_VERSION} --depth 1
  cmake_install \
    -DCRC32C_BUILD_TESTS=OFF \
    -DCRC32C_BUILD_BENCHMARKS=OFF \
    -DCRC32C_USE_GLOG=OFF

  # nlohmann json
  github_checkout nlohmann/json ${NLOHMAN_JSON_VERSION} --depth 1
  cmake_install \
    -DJSON_BuildTests=OFF

  # google-cloud-cpp
  github_checkout googleapis/google-cloud-cpp ${GOOGLE_CLOUD_CPP_VERSION} --depth 1
  cmake_install \
    -DGOOGLE_CLOUD_CPP_ENABLE_EXAMPLES=OFF \
    -DGOOGLE_CLOUD_CPP_ENABLE=storage
}

function install_azure-storage-sdk-cpp {
  # Disable VCPKG to install additional static dependencies under the VCPKG installed path
  # instead of using system pre-installed dependencies.
  export AZURE_SDK_DISABLE_AUTO_VCPKG=ON
  vcpkg_commit_id=7a6f366cefd27210f6a8309aed10c31104436509
  github_checkout azure/azure-sdk-for-cpp azure-storage-files-datalake_${AZURE_SDK_VERSION}
  sed -i "s/set(VCPKG_COMMIT_STRING .*)/set(VCPKG_COMMIT_STRING $vcpkg_commit_id)/" cmake-modules/AzureVcpkg.cmake

  azure_core_dir="sdk/core/azure-core"
  if ! grep -q "baseline" $azure_core_dir/vcpkg.json; then
    # build and install azure-core with the version compatible with system pre-installed openssl
    openssl_version=$(openssl version -v | awk '{print $2}')
    if [[ "$openssl_version" == 1.1.1* ]]; then
      openssl_version="1.1.1n"
    fi
    sed -i "s/\"version-string\"/\"builtin-baseline\": \"$vcpkg_commit_id\",\"version-string\"/" $azure_core_dir/vcpkg.json
    sed -i "s/\"version-string\"/\"overrides\": [{ \"name\": \"openssl\", \"version-string\": \"$openssl_version\" }],\"version-string\"/" $azure_core_dir/vcpkg.json
  fi
  cmake_install $azure_core_dir -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF

  # install azure-storage-common
  cmake_install sdk/storage/azure-storage-common -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF

  # install azure-storage-blobs
  cmake_install sdk/storage/azure-storage-blobs -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF

  # install azure-storage-files-datalake
  cmake_install sdk/storage/azure-storage-files-datalake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF
}
