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

set -efx -o pipefail
function dnf_install {
  dnf install -y -q --setopt=install_weak_deps=False "$@"
}
# https://github.com/googleapis/google-cloud-cpp/blob/main/doc/packaging.md#required-libraries

#dnf or apt based on linux distro
dist=`grep "^ID=" /etc/os-release | awk -F '=' '{print $2}'| sed 's/"//g'`

user=`whoami`
case "$user" in
  root) echo 'Running as root' ; sudocmd="";;
     *) echo 'Running as other user than root' ; sudocmd="sudo";;
esac
if [[ $dist == "ubuntu" ]]; then
  # install google-cloud-cpp dependencies (ubuntu 20.04)
  # https://github.com/googleapis/google-cloud-cpp/blob/main/doc/packaging.md#required-libraries

  export DEBIAN_FRONTEND=noninteractive
  $sudocmd apt-get update && \
  $sudocmd apt install -y apt-transport-https \
      apt-utils \
      automake \
      build-essential \
      ca-certificates \
      curl \
      gcc \
      libc-ares-dev \
      libc-ares2 \
      libcurl4-openssl-dev \
      m4 \
      make \
      pkg-config \
      python3-pip \
      tar \
      wget \
      zlib1g-dev
else
  # install google-cloud-cpp dependencies (Fedora)
  $sudocmd dnf makecache
  $sudocmd dnf_install ccache cmake curl findutils gcc-c++ git make ninja-build \
        openssl-devel patch unzip tar wget zip zlib-devel


  mkdir -p $HOME/Downloads/pkg-config-cpp && cd $HOME/Downloads/pkg-config-cpp
  curl -sSL https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz | \
      tar -xzf - --strip-components=1 && \
      ./configure --with-internal-glib && \
      make -j ${NCPU:-4} && \
  $sudocmd make install && \
  $sudocmd ldconfig

  export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib64/pkgconfig
fi
