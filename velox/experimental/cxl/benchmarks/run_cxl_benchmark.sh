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
# Runs the CXL aggregation benchmark across the placement configurations, each
# in its own process with the appropriate numactl memory policy. The capped
# configs sweep the DRAM limit, and the interleave config sweeps the DRAM:CXL
# placement ratio, so each strategy's sensitivity to its DRAM share is visible:
#
#   A  dram        DRAM pool capped, on-disk spill    (swept over DRAM_MB_LIST)
#   B  interleave  pages striped across DRAM + CXL    (swept over WEIGHTS_LIST)
#   C  cxl         CxlHashAggregation, relocate to CXL (swept over DRAM_MB_LIST)
#
# B's ratio sweep uses weighted interleave (MPOL_WEIGHTED_INTERLEAVE, Linux
# 6.9+, numactl with --weighted-interleave); weights are written to
# /sys/kernel/mm/mempolicy/weighted_interleave/node<N>, which requires root (or
# sudo). Without kernel/numactl support only the classic 1:1 stripe runs.
# Note a Velox-side DRAM cap would NOT control B's ratio: it limits how many
# bytes the query may allocate, not where the kernel places pages.
#
# Environment:
#   BENCH         path to velox_cxl_aggregation_benchmark (required)
#   CXL_NODE      CXL NUMA node id from cxl_numa_setup.sh (required)
#   CXL_MB        CXL pool capacity in MB for config C (default 4096)
#   DRAM_NODE     DRAM/compute NUMA node (default 0)
#   SF            input size, scale_factor = 1 is ~1GB of (k, v) data (default 1).
#                 Tune SF, --zipf_groups (via EXTRA) and the cap sweep together
#                 so the group table exceeds the cap — see the README.
#   DRAM_MB_LIST  DRAM caps in MB to sweep for A and C (default "32 40 48 56",
#                 sized for the default 1M-group table; rescale with zipf_groups)
#   WEIGHTS_LIST  dram:cxl interleave weights to sweep for B (default "1:1")
#   CONFIGS       which configs to run, space-separated subset of "A B C"
#                 (default "A B C"). E.g. CONFIGS="B C" skips the slow spill leg.
#   EXTRA         extra flags passed through (e.g. --zipf_groups, --zipf_skew)
#   RUN_LOG       path for the captured run log (default results/zipf-sf<SF>.log)
#
# All leg output is tee'd to RUN_LOG and a one-line-per-leg summary is printed
# at the end via scripts/summarize.sh.

set -euo pipefail

: "${BENCH:?set BENCH to the benchmark binary path}"
: "${CXL_NODE:?set CXL_NODE to the CXL NUMA node id (see cxl_numa_setup.sh)}"
CXL_MB="${CXL_MB:-4096}"
DRAM_NODE="${DRAM_NODE:-0}"
SF="${SF:-1}"
DRAM_MB_LIST="${DRAM_MB_LIST:-32 40 48 56}"
WEIGHTS_LIST="${WEIGHTS_LIST:-1:1}"
CONFIGS="${CONFIGS:-A B C}"
EXTRA="${EXTRA:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LOG="${RUN_LOG:-${SCRIPT_DIR}/results/zipf-sf${SF}.log}"
mkdir -p "$(dirname "${RUN_LOG}")"

# Word-split EXTRA (e.g. "--zipf_groups=... --zipf_skew=...") into an array.
read -ra extra_flags <<<"${EXTRA}"
common=(--scale_factor="${SF}" "${extra_flags[@]}")

# Returns true when config "$1" is in the requested CONFIGS set.
want_config() {
  [[ " ${CONFIGS} " == *" $1 "* ]]
}

WEIGHT_DIR=/sys/kernel/mm/mempolicy/weighted_interleave
supports_weighted_interleave() {
  [[ -d ${WEIGHT_DIR} ]] && numactl --help 2>&1 | grep -q "weighted-interleave"
}

# Writes a node's interleave weight, via sudo when the sysfs file is not
# directly writable.
set_node_weight() {
  local node="$1" weight="$2"
  local file="${WEIGHT_DIR}/node${node}"
  if [[ -w ${file} ]]; then
    echo "${weight}" >"${file}"
  else
    echo "${weight}" | sudo tee "${file}" >/dev/null
  fi
}

{
  for weights in ${WEIGHTS_LIST}; do
    want_config B || break
    dram_weight="${weights%%:*}"
    cxl_weight="${weights##*:}"
    if [[ ${weights} == "1:1" ]] && ! supports_weighted_interleave; then
      echo "######## B: CXL interleave dram:cxl=1:1 (classic) ########"
      numactl --cpunodebind="${DRAM_NODE}" \
        --interleave="${DRAM_NODE},${CXL_NODE}" \
        "${BENCH}" --config=interleave "${common[@]}"
    elif supports_weighted_interleave; then
      echo "######## B: CXL interleave dram:cxl=${weights} (weighted) ########"
      set_node_weight "${DRAM_NODE}" "${dram_weight}"
      set_node_weight "${CXL_NODE}" "${cxl_weight}"
      numactl --cpunodebind="${DRAM_NODE}" \
        --weighted-interleave="${DRAM_NODE},${CXL_NODE}" \
        "${BENCH}" --config=interleave "${common[@]}"
    else
      echo "######## B: skipping dram:cxl=${weights} — weighted interleave" \
        "needs Linux 6.9+ (${WEIGHT_DIR}) and numactl --weighted-interleave ########"
    fi
  done

  failures=0
  for dram_mb in ${DRAM_MB_LIST}; do
    want_config A || want_config C || break

    if want_config A; then
      echo "######## A: DRAM-only + spill (cap ${dram_mb} MB) ########"
      if ! numactl --cpunodebind="${DRAM_NODE}" --membind="${DRAM_NODE}" \
        "${BENCH}" --config=dram --dram_limit_mb="${dram_mb}" "${common[@]}"; then
        echo ">>> LEG FAILED: dram at ${dram_mb} MB (cap infeasible or error above)"
        failures=$((failures + 1))
      fi
    fi

    want_config C || continue
    echo "######## C: CXL-aware relocate (cap ${dram_mb} MB) ########"
    # The CXL pool binds itself to CXL_NODE via libnuma; the process compute and
    # DRAM stay on DRAM_NODE. A failure at a low cap is a legitimate sweep
    # outcome: the bucket array stays DRAM-pinned, so config C has a hard DRAM
    # floor that config A (which spills the whole table away) does not.
    if ! numactl --cpunodebind="${DRAM_NODE}" --membind="${DRAM_NODE}" \
      "${BENCH}" --config=cxl --cxl_numa_node="${CXL_NODE}" \
      --cxl_capacity_mb="${CXL_MB}" --dram_limit_mb="${dram_mb}" \
      "${common[@]}"; then
      echo ">>> LEG FAILED: cxl at ${dram_mb} MB (below the DRAM bucket-array" \
        "floor, or error above)"
      failures=$((failures + 1))
    fi
  done

  if [[ ${failures} -gt 0 ]]; then
    echo ">>> ${failures} leg(s) failed; see markers above."
  fi
} 2>&1 | tee "${RUN_LOG}"

echo
echo "==== SUMMARY ===="
"${SCRIPT_DIR}/scripts/summarize.sh" "${RUN_LOG}"
