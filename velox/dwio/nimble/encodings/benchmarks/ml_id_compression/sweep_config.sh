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

# shellcheck shell=bash disable=SC2034
# Sweep configuration. Source this from a launcher.
#
# The drivers are general: every reduction here is a flag whose default is the
# old behaviour, so a run without these variables reproduces earlier sweeps.
# The methodology and its justifications live in METHODOLOGY.md beside this
# file; this is only the configuration those arguments produced.

# Row count. RANGE_SIZES below derives its top rung from this, so the two
# cannot drift apart.
SWEEP_ROWS="${SWEEP_ROWS:-524288}"

# --- repeats ----------------------------------------------------------------
# Three, and unpinned. The box's unpinned noise floor is about 0.8%, and pinned
# baseline spreads measured 0.3-1.4% -- the same floor, so taskset was buying
# nothing. The five-repeat, taskset rule belongs to the laptop, where unpinned
# runs are bimodal with a 3.34x spread.
SWEEP_REPEATS=3

# --- arm sets ---------------------------------------------------------------
# Compression and compzl take every arm: one iteration each, so they are cheap,
# and the ablation table needs the full set.
ARMS_ALL="Trivial,FixedBitWidth,Dictionary,RLE,RLE/view,FixedBitWidth/view,\
Dictionary/view,PFOR/view,SimdForBitpack/view,\
FPE/fpe_noindex,FPE/fpe_pertier,FPE/fpe_elias,\
SIS/realNested,SIS/realNested+view,SIS/key_derived,SIS/key_derived+view,\
SIS/auto,SIS/auto+view,\
SIS/huffOn,SIS/huffOn+view,SIS/huffOn/key_derived,SIS/huffOn/key_derived+view,\
openzl/auto"

# The access drivers take the whole SIS family, the OpenZL baseline, the
# FixedBitWidth pair, and the FPE index variants. Fewer arms than compression: the
# grid reductions do not pay for themselves without this, because measurements
# are cells multiplied by arms.
#
# FixedBitWidth/view is the best whitebox arm on every access driver measured:
# best in 20 of 42 gather cells (median 1321.7 Meps), 23 of 42 range cells
# (1381.8) and 12 of 14 skip cells (1309.9). The cursor variant ties on medians
# and takes most of the remaining cells, so both stay -- the pair is the
# view-versus-cursor comparison, not a duplicate.
#
# The FPE index variants are not baseline colour. They are the paper's showcase
# for making a value-grouping encoding randomly addressable, so their access
# numbers are a claim in their own right. fpe_pertier in particular is worst on
# compression in 26 of 26 cells, which twice made it look like the obvious cut,
# and is competitive on point access at 76.9ns against fpe_noindex's 74.7ns.
# fpe_tagtag stays dropped: dominated on all three axes, which is measured and
# is unaffected by keeping the rest of the family.
#
# Dictionary and RLE are absent deliberately. Their access numbers in the
# existing pull came from the defective nested-selection configuration, so they
# would have to be re-measured before they meant anything; FixedBitWidth and
# Trivial have no sub-streams and were never affected, which is what makes the
# best-whitebox finding above safe to build on.
#
# The forced-transform SIS arms stay. Their decode behaviour is a claim in its
# own right: key_derived costs 1.80x on point access on publicbi_npi, the
# compresses-better-probes-worse case that compression numbers alone hide.
ARMS_ACCESS="SIS/realNested,SIS/realNested+view,\
SIS/key_derived,SIS/key_derived+view,\
SIS/auto,SIS/auto+view,\
SIS/huffOn,SIS/huffOn+view,SIS/huffOn/key_derived,SIS/huffOn/key_derived+view,\
openzl/auto,FixedBitWidth,FixedBitWidth/view,\
FPE/fpe_noindex,FPE/fpe_pertier,FPE/fpe_elias"

# The encode driver drops every +view arm. A view is a read path: a +view arm
# encodes the same bytes by the same route as its cursor twin, so measuring
# both measured one of them twice. Across 546 view/cursor pairs in the existing
# enc_* CSVs there were zero payload byte mismatches, with an encode-throughput
# ratio median of 0.993 and 531 of 546 pairs within ten percent; the
# 0.819-1.527 spread is noise on a quantity that should be identical.
#
# If the view ever gains an encode-time component, that equivalence breaks and
# these arms have to come back. See METHODOLOGY.md.
ARMS_ENCODE="SIS/realNested,SIS/auto,SIS/huffOn,SIS/huffOn/key_derived,\
SIS/key_derived,openzl/auto,FixedBitWidth,\
FPE/fpe_noindex,FPE/fpe_pertier,FPE/fpe_elias"

# --- gather ladders ---------------------------------------------------------
# Dense at the bottom: the run-length axis spans 60.45x at selectivity 0.05 and
# 1.01x at 1.0, so small gathers carry the nuance and large ones coalesce.
#
# Both are overridable so a launcher can ask for a different shape grid without
# forking this file; unset, they are exactly the values earlier sweeps used.
GATHER_SELECTIVITY="${GATHER_SELECTIVITY:-0.001,0.01,0.05,0.10,0.33,0.66,1.0}"
# Judgement, not measured -- the pull only contains run lengths 1 and 131072.
GATHER_RUN_LENGTHS="${GATHER_RUN_LENGTHS:-1,111,12417,131072}"

# --- range ladder -----------------------------------------------------------
# The two halves of this list behave differently on purpose.
#
# The lower rungs are absolute sizes and do not scale with N: they exist to
# probe the small-slice regime, and a slice of eight rows is the same
# experiment whatever the column length.
#
# The top rung is the whole column and must track N. Left as a literal, a run
# at 2M would stop measuring the whole-column case entirely -- the one point
# where SIS and OpenZL converge, and therefore the anchor of the decay claim.
RANGE_SIZES="1,8,64,512,4096,32768,131072,${SWEEP_ROWS}"
# Offsets scale with size: the spread across offsets is 1.03x at B=1 and 1.62x
# at B=32768 but 7.44x at B=512, so the middle needs about 32 and the ends need
# only a handful. One entry per RANGE_SIZES entry; the driver refuses a
# mismatch rather than silently pairing them wrongly.
RANGE_OFFSETS_BY_SIZE="8,8,16,32,32,16,8,1"

# --- cache states -----------------------------------------------------------
# Cold is kept. It reads as a null result only if the table is restricted to
# SIS and OpenZL, which are compute-bound; FixedBitWidth/view is memory-bound
# and loses 1.84x cold at selectivity 0.05 with long runs.
ACCESS_CACHE_STATES="hot cold-payload"

# --- encode cache -----------------------------------------------------------
# Set to a directory to share encoded payloads across drivers. Empty disables.
#
# The encode driver ignores this by design -- it would otherwise time the load
# it was supposed to be timing an encode for. See METHODOLOGY.md.
SWEEP_ENCODE_CACHE_DIR="${SWEEP_ENCODE_CACHE_DIR:-}"
