# SubIntSplit encoding

SubIntSplit is a cascading encoding for 32-bit and 64-bit numeric streams. It
partitions each physical value into contiguous bit ranges, encodes each range as
an independent Nimble stream, and reconstructs the original value by shifting
and combining the decoded ranges.

Use SubIntSplit when one integer-sized value contains fields with different
distributions. Common examples include identifiers composed from a timestamp,
a host or shard identifier, and a sequence number. Whole-value encodings see a
single wide integer; SubIntSplit lets each field use the encoding that fits it.

## Supported types

SubIntSplit supports these Nimble physical types:

| Logical type | Physical bit pattern |
|---|---|
| `Int32` | 32-bit signed integer |
| `Uint32` | 32-bit unsigned integer |
| `Int64` | 64-bit signed integer |
| `Uint64` | 64-bit unsigned integer |
| `Float` | Preserved as its 32-bit representation |
| `Double` | Preserved as its 64-bit representation |

Floating-point values are split by physical representation. SubIntSplit does
not reinterpret their exponent or mantissa numerically, and round trips retain
the exact bit pattern, including NaNs.

The writer rejects narrower integers, booleans, strings, and complex values.
Nullability remains an outer concern: a Nullable encoding owns the null stream,
while SubIntSplit receives and encodes the non-null numeric values.

## Architecture at a glance

```text
writer context                                      reader context
serde/layout -> selection -> SubIntSplitEncoding -> encoded bytes
                                  |                       |
                        sample -> plan                    v
                                  |             factory -> SectionTable
                                  v                       |
                           SectionEncoder       child decode -> accumulate -> values
```

Planning happens only on the write path. The serialized section table is the
read contract: it gives each child decoder its bit range and payload. Full and
selective readers use the same child streams and differ only in how they
schedule decoding and gather output rows.

## Runtime contexts

The same encoding participates in several call paths. Identify the active
context before changing a shared helper because each path supplies different
state and exposes different performance costs.

| Context | Entry point | Plan and configuration source | Result |
|---|---|---|---|
| Explicit writer route | `NimbleConfig` → `NimbleWriterOptionBuilder` → `Writer` | `nimble.subintsplit.columns` resolves schema paths; `SubIntSplitEncoding::planSections()` plans unless the layout requests preserve mode | A targeted value stream uses SubIntSplit; other streams retain their existing policy |
| Configured replay | `SubIntSplitEncoding::planSections()` | `subintsplit.mode=preserve` plus serialized boundaries, exclusions, and optional child encodings | Encoding skips the split search and reproduces the supplied layout |
| Full decode | `EncodingFactory` → `SubIntSplitEncoding::materialize()` | Section headers and nested child payloads from the encoded stream | Contiguous values reconstructed in caller-owned output |
| Selective scan | `SubIntSplitEncoding::readWithVisitor()` | Reader selection, filter, null, and hook state | A contiguous span is bulk-decoded when eligible; the fallback refills bounded blocks |
| Random access | `EncodingViewFactory` → `SubIntSplitEncodingView` | View-capable nested child encodings | One logical value reconstructed at a requested row |
| Legacy visitor decode | `legacy/EncodingFactory.cpp` | The same serialized type and physical-width checks as the normal factory | Existing visitor callers receive the same SubIntSplit decoder behavior |

### One value through the format

For input row `i`, the writer treats the numeric value as an unsigned physical
bit pattern. Planning examines sampled rows, while encoding extracts every row
using the final section plan. Child stream `j` stores the bits for section `j`
from every input row, so all children have the same row count and cursor.

The file header records each section range and child payload size. During
decode, `SectionTable` advances every dynamic child by the same number of rows.
`SectionAccumulator` masks each child value to its declared width, shifts it to
the recorded first bit, and combines it with the constant-section accumulator.
The resulting physical bits are copied back to the logical type, preserving
signed values and floating-point payloads exactly.

## Encoding model

For a 64-bit value split into `[0..11]`, `[12..19]`, `[20..27]`, and
`[28..63]`, the logical encoding tree is:

```text
SubIntSplit<Uint64>
├── bits 0..11   → independently selected unsigned child encoding
├── bits 12..19  → independently selected unsigned child encoding
├── bits 20..27  → independently selected unsigned child encoding
└── bits 28..63  → independently selected unsigned child encoding
```

Sections have these invariants:

- They cover the full physical width exactly once.
- They are contiguous, non-overlapping, and ordered from least-significant to
  most-significant bits.
- Each section uses the narrowest unsigned storage type that can hold it:
  `uint8_t`, `uint16_t`, `uint32_t`, or `uint64_t`.
- Nested encoding selection operates independently for each section.
- A constant high prefix or low suffix is represented by a Constant child and
  folded into the decode accumulator once.

The split is structural. Compression may still be applied to nested streams by
their normal compression policies.

## Write path

The write path has four stages.

### 1. Sample the stream

`Sampler.h` converts values to zero-extended physical bit patterns and draws
evenly spaced contiguous windows. Contiguous sampling preserves run structure,
which matters to RLE and frame-like cost estimates.

The default sample contains at most 2,048 values in blocks of 128. Streams
smaller than the limit are sampled in full.

### 2. Build candidate bit ranges

`SplitSelector` first finds the lowest and highest bit positions that vary in
the sample. Constant bits outside that active interval do not enter the dynamic
program and become Constant sections.

Inside the active interval, a position becomes a candidate boundary when the
set rate changes enough between adjacent bit planes. The default relative
threshold is `0.001`. A caller may also impose a hard limit; the selector keeps
the strongest set-rate changes and always retains both active-range edges.

This step bounds the expensive work. If there are `B` retained boundaries and
`S` samples, the cost grid is approximately `O(B² × S)`.

### 3. Score and select a partition

The planner extracts each candidate range from the sample and computes only the
metrics needed by its cost models:

- Minimum, maximum, and range.
- Run count and average run length.
- Exact distinct count up to a fixed cap.
- Dominant-value frequency while the distinct count remains reliable.

Narrow sections use a reusable direct-indexed frequency table. Wider sections
use a capped hash table. Width limits can skip frequency work where Dictionary
and MainlyConstant are unlikely to win.

The cost model estimates Trivial, FixedBitWidth, Constant, MainlyConstant,
Dictionary, RLE, and Varint storage. A dynamic program selects the minimum-cost
partition of the active interval. Each extra section pays:

```text
splitPenalty + decodeCostBitsPerValue × fullValueCount
```

The fixed split penalty accounts for section metadata. The optional per-value
term lets a caller trade storage for decode throughput by discouraging extra
passes over the output.

The planner also records whether sampled evidence rules out RLE or
MainlyConstant for each chosen section. The encoder carries those exclusions
into the full-data nested selector, avoiding expensive estimates that cannot
win. These exclusions apply only to the immediate section; deeper nested
streams retain their normal candidates.

### 4. Encode each section

`SectionEncoder` extracts one range into its narrow unsigned storage type and
runs nested selection on the complete extracted stream. FixedBitWidth receives
the exact section width so a non-byte-aligned range does not round up.

Varint is excluded from automatic section selection because it has no
`EncodingView`; allowing it would make an otherwise valid SubIntSplit stream
unreadable through random access. Explicit preserve-mode configuration can pin
a writable section encoding, while an empty entry delegates that section to
normal nested selection.

The outer stream is assembled only after all child payloads have been produced.
Temporary section buffers can use `Encoding::Options::bufferPool` so repeated
encodes reuse allocation capacity.

## On-disk format

SubIntSplit begins with Nimble's common encoding prefix. Its encoding-specific
payload is:

```text
uint8  number of sections
uint8  flags
repeat number of sections times:
  uint8  first bit, inclusive
  uint8  last bit, inclusive
  uint32 encoded child size
child payload 0
child payload 1
...
```

Section headers and payloads are stored in least-significant-bit order. Each
child payload is a complete Nimble encoding and therefore carries its own type,
row count, and encoding-specific bytes.

Flag bit zero records the optional zigzag-delta pre-transform. Streams written
before the flag was introduced have a zero flags byte and decode as raw values.

Captured layouts serialize three pieces of planner state:

- `subintsplit.mode=preserve` requests replay instead of replanning.
- `subintsplit.boundaries` records inclusive ranges as
  `start-end;start-end;...`.
- `subintsplit.section_candidate_exclusions` retains sampled RLE and
  MainlyConstant pruning decisions.

`subintsplit.section_encodings` can additionally pin a child encoding per
section. Empty entries retain normal nested selection. Boundary parsing rejects
gaps, overlap, out-of-range endpoints, and incomplete coverage.

## Read path

`EncodingFactory` constructs SubIntSplit for all supported physical types. The
legacy visitor dispatcher also recognizes it, so full materialization and
selective reads share the same encoded format.

`SectionTable` parses every child header and creates its nested decoder. It then
classifies the children:

- Constant children are read once and folded into a single shifted value.
- A sole full-width child is a pass-through stream and writes directly to the
  caller's output.
- A sole low-bit Trivial child can widen directly into the output while adding
  any constant prefix.
- Other children decode into reusable scratch storage and are masked, shifted,
  and OR-ed into the output.

The general path works in chunks. Each dynamic child traverses the same output
chunk before the decoder advances, keeping reconstruction data cache-local.
AVX2 builds use vectorized widening and accumulation for narrower children;
other widths and tail values use the scalar loop.

### Selective reads

Integral streams can use the `readWithVisitor` bulk path when the visitor,
filter, null handling, and hardware satisfy Velox's fast-path requirements. It
decodes the contiguous span containing the requested rows once, then gathers
the selected positions. Sparse filters therefore avoid one virtual child decode
per selected value.

Float and double use the general visitor path so conversion from their physical
bit representation remains correct. The slow path refills a small block instead
of calling every child decoder once per value.

### Delta pre-transform

The experimental delta mode encodes the first value verbatim and subsequent
values as zigzag deltas. The writer encodes both raw and delta forms and keeps
the smaller one.

Delta reconstruction requires all preceding values. A delta SubIntSplit stream
therefore supports sequential `materialize()` from row zero, while `skip()` and
`readWithVisitor()` reject it. Keep this option disabled for production until
the format has restart points.

## Selection and configuration

### Explicit writer routing

The safest production entry point is the serde property:

```text
nimble.subintsplit.columns=user_id,nested.event_id,items[*],properties[*]
```

Paths use Velox subfield syntax:

| Shape | Example | Target |
|---|---|---|
| Top-level scalar | `user_id` | The column's value stream |
| Nested row | `nested.event_id` | The nested field's value stream |
| Array | `items[*]` | The array element value stream |
| Map | `properties[*]` | The map value stream |

The property pins the outer SubIntSplit encoding. Its sections still use the
normal planner and nested selection. Every unlisted stream retains the writer's
existing selection behavior.

Writer construction resolves paths against the stored schema and rejects:

- A path that resolves to an unsupported type.
- Duplicate paths that resolve to the same schema node.
- Cluster-index key columns whose regular storage is omitted.
- A value stream also targeted by shared dictionary configuration.
- Custom cluster-index configuration whose key columns cannot be resolved.

An explicit `encodingLayoutTree` is applied after property-based routing and
takes precedence for streams configured by both mechanisms.

### Selection status

The production path in this stack is explicit serde routing. SubIntSplit is a
registered encoding type, while the current default selection factors do not
choose it automatically. The global-candidate benchmark is an experiment used
to compare policies; it is not the rollout behavior documented here.

Explicit routing may produce a one-section encoding when the configured stream
has no profitable split. Benchmark each routed stream because the wrapper adds
metadata without decomposing the value in that case.

## Tuning defaults

`subintsplit::kDefaultTuningConfig` owns the algorithm's production defaults.
They are internal to SubIntSplit instead of fields on the generic
`Encoding::Options` shared by every encoding.

| Setting | Default | Effect |
|---|---:|---|
| Planner samples | 2,048 | Bound values sampled by the planner |
| Boundary prune threshold | `0.001` | Discard weak adjacent bit-plane changes |
| Candidate boundary cap | Unlimited | Bound the split grid when configured by a benchmark |
| Maximum section width | Unlimited | Skip wide grid cells except the full-range fallback |
| Frequency-metrics width | Unlimited | Skip frequency metrics above a width |
| Decode-cost penalty | `0.0` bits/value | Penalize each extra dynamic section |
| Decode chunk | 4,096 values | Bound values reconstructed per decode pass |

Benchmarks and focused tests can pass an alternate `TuningConfig` directly to
`SubIntSplitEncoding`. Normal writer and reader paths always use the constant
default. Planner changes can alter section boundaries and encoded bytes;
decode chunk size does not alter the format.

`Encoding::Options::subIntSplitDeltaPreTransform` remains separate because it
is a runtime rollout gate that changes the stream representation.

## Correctness invariants

Changes to SubIntSplit must preserve these properties:

1. Sections cover every physical bit exactly once and remain ordered.
2. Extracting and recombining sections preserves the input bit pattern.
3. Signed extrema and floating-point NaNs round trip bit-for-bit.
4. Constant-only, one-section, and multi-section plans decode correctly.
5. `reset()`, sequential `materialize()`, `skip()`, and supported visitor paths
   keep every child decoder on the same logical row.
6. Captured layouts reproduce their boundaries and candidate exclusions.
7. Factory and legacy dispatch accept exactly the supported physical widths.
8. Nullable writer streams preserve null positions while SubIntSplit encodes
   only present values.

The test surface includes encoding round trips, format parsing, configured
splits, planner controls, cost models, factory dispatch, random-access views,
writer subfield routing, randomized encoding fuzzing, and deterministic writer
integration across multiple data characteristics.

## Benchmarking and rollout

Use `scripts/suryadev/subintsplit_bench` for production corpus measurements and
`scripts/suryadev/nimble_read_bench` for end-to-end selective-reader workloads.
Run optimized builds and compare identical inputs against current default
auto-selection. Keep correctness verification outside timed regions.

Evaluate storage, encode, and decode independently. A global candidate can save
more space while regressing both CPU axes. Prefer explicit serde routing until
the selector predicts final compressed size and CPU cost accurately enough for
the target workload.

Before enabling a stream:

1. Extract representative partitions and retain the source date and value cap.
2. Compare against current default auto-selection under the intended
   compression policy.
3. Require headroom on storage, encode, and the relevant decode workloads.
4. Verify every encoded stream value-for-value outside the timed loop.
5. Configure only the approved schema paths with
   `nimble.subintsplit.columns`.
6. Re-run the corpus benchmark when the data distribution, compression policy,
   or selection factors change.

## Troubleshooting

- Unexpectedly high encode time usually comes from planner grid size or
  full-data nested estimates. Inspect retained boundaries, sample count, and
  section candidate exclusions.
- Unexpectedly slow decode usually comes from too many dynamic sections or a
  costly nested encoding. Dump the captured layout before changing chunk size.
- A storage regression under MetaInternal can mean whole-value byte patterns
  compress better than independently compressed sections.
- A path configuration error should be corrected at the serde property. Do not
  substitute a positional schema-node identifier.
- A random-access failure indicates that a nested child lacks an
  `EncodingView`; automatic section selection excludes Varint for this reason.

## File guide

Paths below are relative to `dwio/nimble/`.

### Core files: `encodings/subintsplit/`

This directory separates format, planning, encoding, and reconstruction. The
table names every file individually so ownership remains clear when a header
declares an interface and its `.cpp` supplies the policy or algorithm.

| File | Execution context | Inputs and outputs | Contract to preserve |
|---|---|---|---|
| `BitSection.h` | Shared planner/encoder vocabulary | Inclusive first and last bits; produces section ranges and `SectionPlan` entries | Ranges use physical bit positions, remain ordered least-significant first, and report exact widths |
| `CostModel.h` | Planner interface | Section metrics, width, and value count; exposes candidate costs and pruning decisions | Costs use bits consistently and impossible candidates remain distinguishable from expensive ones |
| `CostModel.cpp` | Planner computation | Evaluates Trivial, FixedBitWidth, Constant, MainlyConstant, Dictionary, RLE, and Varint estimates | Estimates and candidate exclusions must derive from the same sampled evidence |
| `DeltaTransform.h` | Optional write transform and sequential read recovery | Physical values ↔ first value plus zigzag residuals | Arithmetic is bit-preserving for signed extrema; transformed streams require decoding from row zero |
| `Format.h` | Persistent write/read boundary | Section count, flags, ranges, child sizes, and payload offsets | Header sizes and field order remain compatible with stored data; each range fits in one-byte endpoints |
| `Sampler.h` | Planner-only preprocessing | Full physical-value span and `SamplerConfig`; produces `uint64_t` samples | Sampling is bounded, deterministic, and block-stratified so local runs survive |
| `SectionAccumulator.h` | Full and selective decode hot path | Decoded unsigned section values, range masks, shifts, and an output span | Scalar and AVX2 paths produce identical physical bits and never leak bits outside a section width |
| `SectionEncoder.h` | Full-data write hot path | One `SectionPlan`, all input values, selection state, and buffer options; produces one child payload | Extraction uses the narrowest safe unsigned type and passes exact bit width plus sampled exclusions to nested selection |
| `SectionMetrics.h` | Planner statistics interface | Declares range, run, cardinality, and dominant-value measurements consumed by cost models | Metrics describe extracted section values rather than whole input values |
| `SectionMetrics.cpp` | Planner statistics implementation | Candidate-range samples and reusable frequency storage; produces `SectionMetrics` | Exact distinct counts stop at the configured cap and scratch state is reset between ranges |
| `SectionTable.h` | Decoder construction and cursor ownership | Serialized section headers and child payloads; produces classified child decoders | Every dynamic child advances by the same logical row count; constants never consume a child cursor |
| `SplitBoundaries.h` | Preserved-layout configuration interface | Declares config keys and parse/serialize APIs for ranges and candidate exclusions | Preserve mode describes a complete physical-width partition |
| `SplitBoundaries.cpp` | Preserved-layout parsing and validation | Text configs ↔ section plans | Rejects gaps, overlaps, malformed endpoints, out-of-range bits, and exclusion-count mismatches |
| `SplitSelector.h` | Planner entry point | Samples, physical width, full row count, and `SelectorConfig`; returns `SelectorResult` | Sentinel defaults preserve standard planning and reported total cost matches the returned plan |
| `SplitSelector.cpp` | Boundary search and dynamic program | Active-bit statistics, retained boundaries, and range costs; produces the minimum-cost partition | Both active-range edges remain candidates, configured caps remain hard limits, and output covers the full width |

### Registration, selection, and writer configuration

| File | Responsibility and reason to change it |
|---|---|
| `encodings/SubIntSplitEncoding.h` | Owns the public encoding implementation. It coordinates planning or layout replay, optional delta comparison, child encoding, serialization, full materialization, skipping, reset, and visitor reads. Start here when behavior spans more than one core helper. |
| `encodings/common/Encoding.h` | Declares SubIntSplit tuning options and their sentinel defaults. Adding a knob starts here and must preserve default behavior. |
| `encodings/common/EncodingFactory.cpp` | Constructs typed SubIntSplit decoders on the normal factory path. |
| `encodings/legacy/EncodingFactory.cpp` | Constructs the same physical types through the legacy visitor dispatch path. |
| `encodings/views/SubIntSplitEncodingView.h` | Implements random access over supported nested section encodings. Change it when adding a view-capable child or adjusting per-row reconstruction. |
| `encodings/views/EncodingViewFactory.cpp` | Registers SubIntSplit with the random-access view factory. A new supported physical type must be wired here as well as in both decoder factories. |
| `encodings/selection/EncodingSelectionPolicy.cpp` | Lists SubIntSplit among registered encoding types. The production defaults in this stack leave selection to explicit writer routing. |
| `writer/WriterOptions.h` | Carries resolved SubIntSplit schema targets into the writer. |
| `writer/Writer.cpp` | Applies explicit targets to value streams and resolves precedence with layout-tree and other stream configuration. |
| `velox/NimbleConfig.*` | Declares and parses `nimble.subintsplit.columns` from serde properties. |
| `writer/fb/NimbleWriterOptionBuilder.cpp` | Transfers the parsed serde configuration into `WriterOptions`. |

### Validation and measurement

| File or directory | Coverage |
|---|---|
| `encodings/tests/SubIntSplitEncodingTest.cpp` | Format round trips, preserved layouts, one-section behavior, and supported physical types. |
| `encodings/tests/SubIntSplitConfiguredSplitTest.cpp` | Explicit boundaries, pinned children, and section-candidate replay. |
| `encodings/tests/SubIntSplitDecodeOptionsTest.cpp` | Decode cost and chunk-size behavior. |
| `encodings/tests/SubIntSplitPlannerOptionsTest.cpp` | Planner sentinels, limits, boundary pruning, and hard-cap invariants. |
| `encodings/tests/SubIntSplitSectionMetricsTest.cpp` | Exact statistics consumed by the cost model. |
| `encodings/tests/SubIntSplitSectionCandidatesTest.cpp` | Per-section candidate exclusion and nested-selection behavior. |
| `encodings/tests/SubIntSplitDeltaTest.cpp` | Delta choice, round trips, and restricted reader operations. |
| `fuzzer/encoding/` | Randomized full decode and `EncodingView` equivalence across supported types and data characteristics. |
| `velox/tests/WriterTest.cpp` | End-to-end serde paths, nested targeting, validation errors, and null preservation. |
| `writer/fb/tests/NimbleWriterOptionBuilderTest.cpp` | Configuration propagation from serde properties to writer options. |
| `encodings/benchmarks/SubIntSplitVsOpenZLBenchmark.cpp` | Focused codec storage, encode, and full-decode comparison. |
| `scripts/suryadev/subintsplit_bench/` | Reproducible production-corpus correctness and performance comparison. |
| `scripts/suryadev/nimble_read_bench/` | End-to-end synthetic projection and selective-filter measurements. |

### Code navigation by change

| Change | Start in | Inspect together | Focused validation |
|---|---|---|---|
| Wire format or flags | `subintsplit/Format.h` and `SubIntSplitEncoding::writeEncoding()` | `SectionTable.h`, legacy factory, and old-format compatibility | Encoding and configured-split tests |
| Boundary or partition heuristic | `SplitSelector.*` | `Sampler.h`, `SectionMetrics.*`, and `CostModel.*` | Planner-option and section-metrics tests; optimized encode benchmark |
| Nested child choice | `SectionEncoder.h` | `CostModel.*`, selection policy, and view support | Section-candidate, encoding-view, and production-corpus tests |
| Full-decode performance | `SectionTable.h` | `SectionAccumulator.h` and `SubIntSplitEncoding::materialize()` | Encoding fuzzer and full-decode benchmarks |
| Filter or projection performance | `SubIntSplitEncoding::readWithVisitor()` | `bulkScan()`, pending-block state, and `SubIntSplitEncodingView.h` | View fuzzer and selective-reader benchmark |
| Serde property behavior | `velox/NimbleConfig.*` | option builder, `WriterOptions.h`, and `Writer.cpp` | Writer and option-builder tests |
| Selection eligibility | `EncodingSelectionPolicy.cpp` | Production factors and explicit layouts | Production-corpus comparison against the current default selector |
| Delta transform | `DeltaTransform.h` | format flags and sequential reader restrictions | Delta tests plus full sequential round trips |
