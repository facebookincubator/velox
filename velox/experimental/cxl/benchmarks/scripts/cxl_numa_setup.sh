#!/usr/bin/env bash
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
#
# Bring a CXL memory expander online as a CPU-less (memory-only) NUMA node, so
# it can be targeted with `numactl --membind` (config C's DRAM pool stays on
# node 0 while the CXL pool binds itself via libnuma) and with
# `numactl --interleave=0,<node>` (config B). The same onlined node serves both
# policies — they are different numactl flags over the same node, not different
# setups.
#
# Requires root and daxctl (the ndctl package). The resulting NUMA node id
# varies by machine and is printed at the end; pass it to the benchmark as
# --cxl_numa_node and to run_cxl_benchmark.sh as CXL_NODE.
#
# Usage: sudo ./cxl_numa_setup.sh [dax_device]   (default: dax0.0)

set -euo pipefail

DAX_DEVICE="${1:-dax0.0}"

if ! command -v daxctl >/dev/null 2>&1; then
  echo "daxctl not found; install ndctl (Debian/Ubuntu: ndctl, RHEL: ndctl)." >&2
  exit 1
fi

echo "== Current dax devices =="
daxctl list || true

echo
echo "== Reconfiguring ${DAX_DEVICE} to system-ram (CPU-less NUMA node) =="
# system-ram onlines the device's memory as a regular kernel-managed NUMA node.
# A CXL expander has no local CPUs, so the node comes up CPU-less.
daxctl reconfigure-device --mode=system-ram --force "${DAX_DEVICE}"

echo
echo "== NUMA topology (look for a node with memory but no cpus) =="
numactl -H

echo
echo "== CPU-less nodes (candidate CXL nodes) =="
# A node listed in /sys with memory but absent from numactl's "cpus" lines is
# the CXL node. Print node ids whose cpulist is empty.
for node in /sys/devices/system/node/node[0-9]*; do
  id="${node##*/node}"
  cpulist="$(cat "${node}/cpulist" 2>/dev/null || true)"
  if [[ -z ${cpulist} ]]; then
    echo "  node ${id}: CPU-less (likely CXL) -> use --cxl_numa_node=${id}"
  fi
done
