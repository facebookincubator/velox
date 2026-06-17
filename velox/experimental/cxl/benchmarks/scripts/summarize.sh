#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Distill a run_cxl_benchmark.sh log into a one-line-per-leg table. The raw log
# interleaves the per-leg results with arbitrator MEM_CAP_EXCEEDED stack dumps
# (expected on config C under a tight cap — they are the relocation trigger),
# which makes the numbers that matter hard to find. This prints just those.
#
# Usage:
#   summarize.sh [log-file]     # defaults to reading stdin
#   run_cxl_benchmark.sh ... | summarize.sh

set -euo pipefail

awk '
  # A leg is bounded by the "config=..." line and its trailing "result:" line.
  /^config=/ {
    config = ""; sf = ""; cap = ""; median = ""; peak = "";
    reloc = "-"; spill = "-";
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^config=/)         { split($i, a, "="); config = a[2] }
      if ($i ~ /^scale_factor=/)   { split($i, a, "="); sf = a[2] }
      if ($i ~ /^dram_limit_mb=/)  { split($i, a, "="); cap = a[2] }
    }
    have = 1; next
  }
  /^median elapsed:/ { median = $3; next }
  /^aggregation:/ {
    for (i = 1; i <= NF; i++) if ($i ~ /^peak=/) { split($i, a, "="); peak = a[2] }
    next
  }
  /^cxl relocations:/ { reloc = $3; next }
  /^spill: bytes=/    { split($2, a, "="); spill = a[2] }   # bytes=NNN.N
  /^LEG FAILED/       { print "  !! " $0 }
  /^result:/ && have {
    printf "%-11s SF%-4s cap=%-6s  median=%8s ms  peak=%9s MB  reloc=%-3s  spill=%-8s\n",
      config, sf, (cap == "" ? "-" : cap "MB"), median, peak, reloc, spill;
    have = 0
  }
' "${@:-/dev/stdin}"
