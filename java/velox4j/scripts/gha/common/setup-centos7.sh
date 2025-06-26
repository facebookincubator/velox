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

# Setup YUM.
sed -i -e "s/enabled=1/enabled=0/" /etc/yum/pluginconf.d/fastestmirror.conf
sed -i -e "s|mirrorlist=|#mirrorlist=|g" /etc/yum.repos.d/CentOS-*
sed -i -e "s|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g" /etc/yum.repos.d/CentOS-*
yum install -y centos-release-scl
rm -f /etc/yum.repos.d/CentOS-SCLo-scl.repo
sed -i \
  -e 's/^mirrorlist/#mirrorlist/' \
  -e 's/^#baseurl/baseurl/' \
  -e 's/mirror\.centos\.org/vault.centos.org/' \
  /etc/yum.repos.d/CentOS-SCLo-scl-rh.repo

# Install essentials.
yum -y install epel-release
yum -y install wget curl tar zip unzip which patch sudo \
  ninja-build perl-IPC-Cmd autoconf autoconf-archive automake libtool \
  devtoolset-11 python3 pip dnf \
  bison java-1.8.0-openjdk java-1.8.0-openjdk-devel \
  ccache \
  lz4-devel lzo-devel libzstd-devel snappy-devel double-conversion-devel \
  libevent-devel

# Link cc / c++ to the ones in devtoolset.
rm -f /usr/bin/cc /usr/bin/c++ /usr/bin/ld
ln -s /opt/rh/devtoolset-11/root/usr/bin/cc /usr/bin/cc
ln -s /opt/rh/devtoolset-11/root/usr/bin/c++ /usr/bin/c++
ln -s /opt/rh/devtoolset-11/root/usr/bin/ld /usr/bin/ld
cc --version
c++ --version
ld --version

pip3 install --upgrade pip

# Install CMake >= 3.28.3.
pip3 install cmake==3.28.3

# Install Git >= 2.7.4
case "$(git --version)" in "git version 2."*)
  true
  ;;
*)
  [ -f /etc/yum.repos.d/ius.repo ] || yum -y install https://repo.ius.io/ius-release-el7.rpm
  yum -y remove git
  yum -y install git236
  ;;
esac

# Install OpenSSL >= 1.1.1.
cd /tmp
wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1o/openssl-1.1.1o.tar.gz
tar -xzvf openssl-1.1.1o.tar.gz
cd openssl-1.1.1o
./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic
make
make install

# Install FLEX >= 2.6.0.
case "$(PATH="/usr/local/bin:$PATH" flex --version 2>&1)" in "flex 2.6."*)
  true
  ;;
*)
  cd /tmp
  yum -y install gettext-devel
  FLEX_URL="https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz"
  mkdir -p flex
  wget -q --max-redirect 3 -O - "$FLEX_URL" | tar -xz -C flex --strip-components=1
  cd flex
  ./autogen.sh
  ./configure
  make install
  ;;
esac

# Install ICU 72.1.
cd /tmp
wget https://github.com/unicode-org/icu/releases/download/release-72-1/icu4c-72_1-src.tgz
tar -xzvf icu4c-72_1-src.tgz
cd icu
source/configure --prefix=/usr --libdir=/usr/lib64 --disable-tests --disable-samples
make
make install

# Install Java 11.
cd /tmp
yum install -y java-11-openjdk-devel
alternatives --set java java-11-openjdk.x86_64

# Install Maven.
if [ -z "$(which mvn)" ]; then
  MAVEN_VERSION=3.9.2
  MAVEN_INSTALL_DIR=/opt/maven-$MAVEN_VERSION
  if [ -d /opt/maven-$MAVEN_VERSION ]; then
    echo "Failed to install maven: ${MAVEN_INSTALL_DIR} already exists." >&2
    exit 1
  fi

  cd /tmp
  wget https://archive.apache.org/dist/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz
  tar -xvf apache-maven-$MAVEN_VERSION-bin.tar.gz
  rm -f apache-maven-$MAVEN_VERSION-bin.tar.gz
  mv apache-maven-$MAVEN_VERSION "${MAVEN_INSTALL_DIR}"
  ln -s "${MAVEN_INSTALL_DIR}/bin/mvn" /usr/local/bin/mvn
fi

# Install patchelf.
cd /tmp
mkdir patchelf
cd patchelf
wget https://github.com/NixOS/patchelf/releases/download/0.17.2/patchelf-0.17.2-x86_64.tar.gz
tar -xvf patchelf-0.17.2-x86_64.tar.gz
ln -s /tmp/patchelf/bin/patchelf /usr/local/bin/patchelf
patchelf --version
