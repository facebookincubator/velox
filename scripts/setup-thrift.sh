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
source ${SCRIPTDIR}/setup-helper-functions.sh

# Propagate errors and improve debugging.
set -efx -o pipefail

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}

function install_thrift {
  local THRIFT_REPO_NAME="apache/thrift"
  local THRIFT_VERSION="0.16.0"

  github_checkout $THRIFT_REPO_NAME $THRIFT_VERSION --depth 1 --recurse-submodules

  ./bootstrap.sh
  ./configure --without-perl --without-python --without-py3 --without-ruby --without-swift
  make
  sudo make install
}

cd "${DEPENDENCY_DIR}" || exit

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
   yum -y install gcc-c++
fi

if [[ "$OSTYPE" == darwin* ]]; then
   brew install libtool automake autoconf QT5 pkg-config
   export PATH=/usr/local/opt/bison/bin:$PATH
   export LDFLAGS="-L/usr/local/opt/bison/lib $LDFLAGS"
   export LDFLAGS="-L/usr/local/opt/openssl@3/lib $LDFLAGS"
   export CPPFLAGS="-Wno-inconsistent-missing-override -I/usr/local/opt/openssl@3/include $CPPFLAGS"
fi

#install_libevent
install_thrift

_ret=$?
if [ $_ret -eq 0 ] ; then
   echo "All deps for Velox thrift installed!"
fi
