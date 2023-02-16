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

# Minimal setup for Amazon Linux 2
set -eufx -o pipefail
SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh

CPU_TARGET="${CPU_TARGET:-avx}"

COMPILER_FLAGS=$(get_cxx_flags "$CPU_TARGET")
export COMPILER_FLAGS

export CXXFLAGS=$COMPILER_FLAGS  # Used by boost.
export CPPFLAGS=$COMPILER_FLAGS  # Used by LZO.

FB_OS_VERSION=v2022.11.14.00
NPROC=$(getconf _NPROCESSORS_ONLN)
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}

GCC_FROM_SOURCE=0

# Install all velox dependencies.
sudo yum install -y \
    ninja-build \
    git \
    openssl11-devel \
    bzip2-devel \
    bzip2 \
    libevent \
    libevent-devel \
    libicu \
    libicu-devel \
    lz4 \
    lz4-devel \
    lz4-static \
    libzstd \
    libzstd-devel \
    libzstd-static \
    zstd \
    snappy \
    snappy-devel \
    lzo \
    lzo-devel \
    bison \
    tzdata \
    wget

# sudo_cmake_install compile and install a dependency
#
# This function uses cmake and ninja-build to compile and install
# a specified dependency. The caller is responsible for making sure
# that the code has been checked out and the current folder contains
# it.
#
# This function requires elevated privileges for the install part
function sudo_cmake_install {
  local NAME=$(basename "$(pwd)")
  local BINARY_DIR=_build
  CFLAGS=$(get_cxx_flags "$CPU_TARGET")

  if [ -d "${BINARY_DIR}" ] && prompt "Do you want to rebuild ${NAME}?"; then
    rm -rf "${BINARY_DIR}"
  fi
  mkdir -p "${BINARY_DIR}"
  CPU_TARGET="${CPU_TARGET:-avx}"

  # CMAKE_POSITION_INDEPENDENT_CODE is required so that Velox can be built into dynamic libraries \
  cmake -Wno-dev -B"${BINARY_DIR}" \
    -GNinja \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 \
    "${INSTALL_PREFIX+-DCMAKE_PREFIX_PATH=}${INSTALL_PREFIX-}" \
    "${INSTALL_PREFIX+-DCMAKE_INSTALL_PREFIX=}${INSTALL_PREFIX-}" \
    -DCMAKE_CXX_FLAGS="$CFLAGS" \
    -DBUILD_TESTING=OFF \
    "$@"

  # install a prebuilt project with elevated privileges
  # This is useful for systems where /usr/ is not writable
  local BINARY_DIR=_build
  sudo ninja-build -C "${BINARY_DIR}" install
}

function clean_dir {
  local DIRNAME=$(basename $1)
  if [ -d "${DIRNAME}" ]; then
    rm -rf "${DIRNAME}"
  fi
  mkdir ${DIRNAME}
}

function run_and_time {
  time "$@"
  { echo "+ Finished running $*"; } 2> /dev/null
}

function git_clone {
  local NAME=$1
  shift
  local REPO=$1
  shift
  local GIT_CLONE_PARAMS=$@
  local DIRNAME=$(basename $NAME)
  if [ -d "${DIRNAME}" ]; then
    rm -rf "${DIRNAME}"
  fi
  git clone -q $GIT_CLONE_PARAMS "${REPO}"
}

function prompt {
  (
    while true; do
      local input="${PROMPT_ALWAYS_RESPOND:-}"
      echo -n "$(tput bold)$* [Y, n]$(tput sgr0) "
      [[ -z "${input}" ]] && read input
      if [[ "${input}" == "Y" || "${input}" == "y" || "${input}" == "" ]]; then
        return 0
      elif [[ "${input}" == "N" || "${input}" == "n" ]]; then
        return 1
      fi
    done
  ) 2> /dev/null
}

function install_checkinstall {
  if [[ -v SKIP_CHECKINSTALL ]]
  then
      return 0
  fi

  local DIRNAME=$(basename "checkinstall")
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  git_clone checkinstall http://checkinstall.izto.org/checkinstall.git
  cd checkinstall
  make
  sudo make install

  popd
}

function install_cmake {
  if [[ -v SKIP_CMAKE ]]
  then
      return 0
  fi

  local DIRNAME=$(basename "cmake-3.25.1")
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh
  chmod +x cmake-3.25.1-linux-x86_64.sh
  sudo ./cmake-3.25.1-linux-x86_64.sh --prefix=/usr/local/ --skip-license

  popd
}

function install_ccache {
  if [[ -v SKIP_CCACHE ]]
  then
      return 0
  fi
  local DIRNAME=$(basename "ccache-for-velox")
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  wget https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz
  tar -xf ccache-4.7.4-linux-x86_64.tar.xz
  sudo cp ccache-4.7.4-linux-x86_64/ccache /usr/local/bin/

  popd
}

function install_flex26 {
  if [[ -v SKIP_FLEX ]]
  then
      return 0
  fi

  local DIRNAME=$(basename "flex-for-velox")
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  wget https://github.com/westes/flex/files/981163/flex-2.6.4.tar.gz
  tar -xzvf flex-2.6.4.tar.gz
  cd flex-2.6.4
  ./configure
  make
  sudo make install
  
  popd
}

function install_boost {
  if [[ -v SKIP_BOOST ]]
  then
      return 0
  fi

  local DIRNAME=$(basename "boost-for-velox")
  clean_dir ${DIRNAME}

  pushd ./

  cd ${DIRNAME}

  local BOOST_VERSION=boost_1_81_0
  wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/${BOOST_VERSION}.tar.bz2
  bunzip2 ${BOOST_VERSION}.tar.bz2
  tar -xf ${BOOST_VERSION}.tar
  cd ${BOOST_VERSION}
  ./bootstrap.sh
  sudo env PATH=${PATH} ./b2 install

  popd
}

function install_gcc {
  if [[ -v SKIP_GCC ]] || [[ ${GCC_FROM_SOURCE} == 0 ]]
  then
      return 0
  fi

  # setup various dependencies needed for building gcc
  sudo yum install -y \
    gcc \
    gcc-c++ \
    binutils \
    gmp \
    mpfr \
    mpfr-devel \
    libmpc \
    libmpc-devel \
    isl \
    isl-devel \
    libzstd \
    libzstd-devel \
    gettext-lib \
    gettext-devel \
    gperf \
    gperftools \
    gperftools-libs \
    gperftools-devel \
    dejagnu \
    expect \
    expect-devel \
    tcl \
    tcl-devel \
    autogen \
    guile \
    guile-devel \
    glibc \
    glibc-all-langpacks \
    glibc-common


  # do all the work in a folder inaptly called gcc-for-velox
  DIRNAME=gcc-for-velox
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  # download gcc code
  wget https://mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-9.4.0/gcc-9.4.0.tar.gz
  tar -zxf gcc-9.4.0.tar.gz
  cd gcc-9.4.0
  mkdir build
  cd build

  ../configure \
    --with-pkgversion=gcc-9.4.0-velox-al2 \
    --host=x86_64-pc-linux-gnu \
    --prefix=/usr/ \
    --libdir=/usr/lib \
    --enable-shared \
    --enable-threads \
    --disable-multilib \
    --enable-tls

  # speed up the build (which can take up to 3 hours)
  # by parallelizing the work
  cpus=$(grep -c ^processor /proc/cpuinfo)
  make -j ${cpus}

  sudo make install

  popd
}

function install_gcc_from_rpm {
  if [[ -v SKIP_GCC_RPM ]]
  then
      return 0
  fi

  # do all the work in a folder inaptly called gcc-for-velox
  DIRNAME=gcc-for-velox
  clean_dir ${DIRNAME}
  pushd ./
  cd ${DIRNAME}

  sudo yum-config-manager --add-repo http://mirror.centos.org/centos/7/sclo/x86_64/rh/

  HAS_LIBGFORTRAN=$(rpm -q --quiet libgfortran5 || echo "missing")

  if [ "$HAS_LIBGFORTRAN" == 'missing' ]; then
    wget http://mirror.centos.org/centos/7/os/x86_64/Packages/libgfortran5-8.3.1-2.1.1.el7.x86_64.rpm
    sudo yum install -y libgfortran5-8.3.1-2.1.1.el7.x86_64.rpm
  fi

  sudo yum install -y devtoolset-11 --nogpgcheck

  source /opt/rh/devtoolset-11/enable

  popd
}

function install_doubleconversion {
  pushd ./
  github_checkout google/double-conversion v3.2.0
  sudo_cmake_install
  popd
}

function install_glog {
  pushd ./
  github_checkout google/glog v0.6.0
  sudo_cmake_install
  popd
}

function install_gflags {
  pushd ./
  github_checkout gflags/gflags v2.2.2
  sudo_cmake_install -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=ON -DINSTALL_HEADERS=ON
  popd
}

function install_gmock {
  pushd ./
  github_checkout google/googletest release-1.12.1
  sudo_cmake_install
  popd
}

function install_re2 {
  pushd ./
  github_checkout google/re2 2022-12-01
  sudo_cmake_install
  popd
}

function install_conda {
  pushd ./
  local MINICONDA_PATH=$(basename "miniconda-for-velox")

  cd ${DEPENDENCY_DIR}
  clean_dir "conda"
  pushd ./
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh -b -p $MINICONDA_PATH
  popd
  popd
}

function install_protobuf {
  pushd ./
  github_checkout protocolbuffers/protobuf v3.21.4
  git submodule update --init --recursive
  sudo_cmake_install -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/lib -Dprotobuf_BUILD_TESTS=OFF
  popd
}

function install_velox_deps {
  echo "Using ${DEPENDENCY_DIR} for installing dependencies"

  mkdir -p ${DEPENDENCY_DIR}

  # the following dependencies are managed by
  # 3rd party repositories
  pushd ./
  cd ${DEPENDENCY_DIR}

  run_and_time install_gcc_from_rpm
  run_and_time install_gcc
  run_and_time install_cmake
  run_and_time install_ccache
  run_and_time install_checkinstall
  run_and_time install_flex26
  run_and_time install_boost
  popd

  # the following dependencies are managed by
  # github repositories
  run_and_time install_doubleconversion
  run_and_time install_glog
  run_and_time install_gflags
  run_and_time install_gmock
  run_and_time install_re2
  run_and_time install_conda
}

(return 2> /dev/null) && return # If script was sourced, don't run commands.

(
  if [[ $# -ne 0 ]]; then
    for cmd in "$@"; do
      run_and_time "${cmd}"
    done
  else
    install_velox_deps
  fi
)

echo "All deps for Velox installed!"
echo "If gcc was installed from a 3rd party rpm (default), execute"
echo "$ source /opt/rh/devtoolset-11/enable"
echo "to setup the correct paths then execute make to compile velox"
