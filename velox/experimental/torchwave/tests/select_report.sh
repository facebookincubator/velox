#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <data_directory>"
    exit 1
fi

DATA_DIR="$(realpath "$1")"
TARGET="fbcode//velox/experimental/torchwave/tests:executor_test"

NS=(1 2 4 8 16 32 64 128)
SIZES=(1000 2000 4000 8000 16000 32000 64000 128000 256000 512000 1024000)

printf "%-4s  %8s  %15s  %15s  %15s  %15s\n" "n" "size" "serial_gpu_p90" "wave_p90" "wave_cg_p90" "wave_1blk_p90"
printf "%-4s  %8s  %15s  %15s  %15s  %15s\n" "----" "--------" "---------------" "---------------" "---------------" "---------------"

for n in "${NS[@]}"; do
    for size in "${SIZES[@]}"; do
        name="select_${n}_${size}"
        custom_path="${DATA_DIR}/${name}"

        if [[ ! -f "${custom_path}.pt2" ]]; then
            echo "SKIP: ${custom_path}.pt2 not found"
            continue
        fi

        # Run 1: multi-block, serial GPU + wave
        output1=$(buck run @//mode/opt "${TARGET}" -- \
            --gtest_filter="*.custom" \
            --custom "${custom_path}" \
            --single_block 0 \
            --num_repeats 40 2>&1) || true

        serial_p90=$(echo "$output1" | grep "serial GPU.*repeats.*p90=" | sed 's/.*p90=\([0-9]*\).*/\1/' | head -1)
        wave_p90=$(echo "$output1" | grep " wave ([0-9]* repeats).*p90=" | sed 's/.*p90=\([0-9]*\).*/\1/' | head -1)

        # Run 2: cg mode, wave only
        output2=$(buck run @//mode/opt "${TARGET}" -- \
            --gtest_filter="*.custom" \
            --custom "${custom_path}" \
            --wave_only \
            --cg 1 \
            --num_repeats 40 2>&1) || true

        wave_cg_p90=$(echo "$output2" | grep " wave ([0-9]* repeats).*p90=" | sed 's/.*p90=\([0-9]*\).*/\1/' | head -1)

        # Run 3: single block, wave only
        output3=$(buck run @//mode/opt "${TARGET}" -- \
            --gtest_filter="*.custom" \
            --custom "${custom_path}" \
            --wave_only \
            --single_block 1 \
            --num_repeats 40 2>&1) || true

        wave_1blk_p90=$(echo "$output3" | grep " wave ([0-9]* repeats).*p90=" | sed 's/.*p90=\([0-9]*\).*/\1/' | head -1)

        printf "%-4s  %8s  %15s  %15s  %15s  %15s\n" \
            "$n" "$size" \
            "${serial_p90:-N/A}" "${wave_p90:-N/A}" "${wave_cg_p90:-N/A}" "${wave_1blk_p90:-N/A}"
    done
done
