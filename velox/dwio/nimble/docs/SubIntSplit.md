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
                                  |             factory -> parseSections
                                  v                       |
                          encodeResiduals       child decode -> accumulate -> values
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
decode, `SubIntSplitEncoding` advances every dynamic child by the same number
of rows.
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

Inside the active interval, a position is a candidate boundary when the set
rate changes enough between adjacent bit planes; the threshold defaults to
0.001, and `TuningConfig::selector.boundaryPruneThreshold` overrides it (0.0 keeps every
position). A caller may also impose a
hard limit; the selector keeps the strongest set-rate changes and always
retains both active-range edges. `selector.trimConstantPlanes`, on by
default, stores constant edge planes as Constant sections before the grid is
scored.

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

The cost models mirror section selection's own size estimators, so the plan
the planner prices is the one nested selection will realise. They cover
Trivial, FixedBitWidth, Constant, MainlyConstant, Dictionary, RLE, Varint,
SimdForBitpack, PFOR, BlockBitPacking, Delta, FOR, and FrequencyPartition.
Huffman and DeltaBlock are priced only when `selector.allowHuffman` or
`selector.allowDeltaBlock` is set, because section selection does not offer
them. `TuningConfig::allowedEncodings` narrows the set further. A dynamic program
selects the minimum-cost partition of the active interval. Each extra section
pays:

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

`SubIntSplitEncoding::encodeResiduals()` extracts each range into its narrow
unsigned storage type and runs nested selection on the complete extracted
stream, under the options `sectionEncodingOptions()` derives. FixedBitWidth
receives the exact section width so a non-byte-aligned range does not round up.

A section's candidates come from `nestedEncodingReadFactors()` in
`EncodingSelectionPolicy.h`, which adds SimdForBitpack, PFOR, BlockBitPacking,
Delta, FOR, and FrequencyPartition to the parent's list for a SubIntSplit
parent. The planner and section selection must agree on this list.

Varint is excluded from automatic section selection because it has no
`EncodingView`; allowing it would make an otherwise valid SubIntSplit stream
unreadable through random access. Explicit preserve-mode configuration can pin
a writable section encoding, while an empty entry delegates that section to
normal nested selection.

The outer stream is assembled only after all child payloads have been produced.
Temporary section buffers can use `Encoding::Options::bufferPool` so repeated
encodes reuse allocation capacity. With `TuningConfig::sectionExecutor` set,
sections are encoded concurrently, each into its own buffer.

### 5. Hold the plan to a whole-value floor

A written stream is never larger than one whole-value section. After planning,
`WholeValueFloor` compares the plan with FixedBitWidth's exact size and with
section selection's pick for the whole value, priced on eight spread sample
blocks. A candidate is encoded only when its estimate, divided by a
per-encoding slack for estimators known to over-quote, undercuts the plan.
The pick is never Delta or Varint, which replay every earlier row to reach
one, except that a plan whose sections are all constant but one stored in
Delta or Varint is also compared with the whole value in that encoding.
When the sample's quote is well under the planner's own estimate, the
fallback is priced first and the plan is abandoned at the first section whose
written bytes already lose. Replayed (preserve-mode) layouts are left alone.

### Decode costs and the hybrid planner

Both are off by default and leave the plan unchanged at their defaults.

- `TuningConfig::selector.decodeWeighting` adds a per-encoding decode cost from
  `DecodeCost.h`, for its access pattern and read path, to the planner's DP.
  `sectionEncodingOptions` carries the same weighting to each section's own
  encoding selection through `Encoding::Options::subIntSplit`. Every
  decode-weighted choice is bounded on size by `TuningConfig::maxSizeRegression`
  against the size-only choice, and the whole-value floor admits the same
  regression.
- `TuningConfig::hybridPlanner` uses the DP only as a shortlister: it re-prices
  the best plans with section selection's estimators on the planner sample
  (`PlanRefiner`), then moves boundaries locally while the price improves.

## On-disk format

SubIntSplit begins with Nimble's common encoding prefix. Its encoding-specific
payload is:

```text
uint8  number of sections
uint8  flags
only when flag bit one is set (row frame):
  uint8  guard, 0xFE
  uint64 slope
  uint64 base
only when flag bit two is set (section transforms):
  uint8  key section, or 0xFF for none
  repeat number of sections times:
    uint8  transform id, 0 for none
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

Flag bit zero records the optional zigzag-delta pre-transform. Flag bit one
records a row frame and announces the 17-byte frame block after the flags byte.
Flag bit two announces the transform block, and is set only on a stream written
as `SubIntSplitReordered` (encoding type 27), so a reader without transform
support fails in the factory instead of returning permuted values. A reader
rejects any flag it does not know, an encoding type that disagrees with the
transform block, an unknown transform id, and a delta stream that also carries
a frame or transforms. Streams written before the flags were introduced have a
zero flags byte and decode as raw values; a stream with no frame is
byte-identical to one written before frames existed.

Captured layouts serialize three pieces of planner state:

- `subintsplit.mode=preserve` requests replay instead of replanning.
- `subintsplit.boundaries` records inclusive ranges as
  `start-end;start-end;...`.
- `subintsplit.section_candidate_exclusions` retains sampled RLE and
  MainlyConstant pruning decisions.
- `subintsplit.row_frame=1` records that the stream carried a row frame, so a
  replay plans the same boundaries over the same residuals. Absent means no
  frame.

`subintsplit.section_encodings` can additionally pin a child encoding per
section. Empty entries retain normal nested selection. Boundary parsing rejects
gaps, overlap, out-of-range endpoints, and incomplete coverage.

## Read path

`EncodingFactory` constructs SubIntSplit for all supported physical types. The
legacy visitor dispatcher also recognizes it, so full materialization and
selective reads share the same encoded format.

`parseSections()` in `Format.h` parses the stream header and every child
header. It rejects, with `NIMBLE_CHECK_FILE`, unknown flags, sections that do
not tile the value width, child sizes that do not account for the whole stream,
and more sections than value bits. The constructor also checks each child's
data type against its bit width, then classifies the children:

- Constant children are read once and folded into a single shifted value
  (`TuningConfig::foldConstantSections`, on by default).
- A sole verbatim child is a pass-through stream and writes directly to the
  caller's output (`TuningConfig::passThrough`, on by default).
- A sole low-bit Trivial child can widen directly into the output while adding
  any constant prefix.
- Other children decode into reusable scratch storage and are masked, shifted,
  and OR-ed into the output.

The general path works in chunks of at most `TuningConfig::decodeChunkSize`
values, with scratch sized to one chunk. Each dynamic child traverses the same
output chunk before the decoder advances, keeping reconstruction data
cache-local.
AVX2 builds use vectorized widening and accumulation for narrower children;
other widths and tail values use the scalar loop.

### Selective reads

Integral streams can use the `readWithVisitor` bulk path when the visitor,
filter, null handling, and hardware satisfy Velox's fast-path requirements. It
decodes the contiguous span containing the requested rows once, then gathers
the selected positions. Sparse filters therefore avoid one virtual child decode
per selected value.

Float and double use the general visitor path so conversion from their physical
bit representation remains correct. With `TuningConfig::visitorBlockBuffer`, on by
default, the slow path refills a small block instead of calling every child
decoder once per value.

### Random access through the view

`createEncodingView` returns `SubIntSplitEncodingView` for `SubIntSplit` and
`SubIntSplitReordered` streams, framed or not. Only a delta stream goes to
`DecodedFallbackEncodingView`, which decodes it once, since a delta value depends
on every row before it. The view holds one `EncodingView` per non-constant
section, falling back to a decoded copy for a section whose encoding has no
view, and folds constant sections into one shifted value.

- A point read ORs one probe per section and adds the row frame back.
- A range read decodes each section in chunks of 1,024 rows into stack
  scratch and accumulates it into the output, so a view keeps no mutable state
  and can be read from several threads.
- A range-list read groups ranges whose gaps cost less than a separate probe
  per row and decodes each group's covering span once. Section views receive
  their own range lists through the checked `EncodingView::read`.

A reordered stream needs, for each row, the position its permuted sections
stored it at. The view builds that position map once per view and caches it
per thread, keyed on an id no later view reuses. It reads the key section in
one bulk call and numbers the key's distinct values through a table indexed
by value when the key is at most 20 bits wide. A wider key leaves the
numbering to the transform, which hashes. With the map, a read picks one of
three plans by length:

- Below 24 rows, the permuted sections are probed through the map.
- Up to half the column, the requested rows' source positions are radix-sorted
  so each section is read as one ascending range list.
- Beyond that, the column is decoded once and the rows are copied out.

A range list is priced range by range against the same three plans, and the
whole list either takes the span plan or decodes the column once.

### Delta pre-transform

The experimental delta mode encodes the first value verbatim and subsequent
values as zigzag deltas. The writer encodes both raw and delta forms and keeps
the smaller one.

Delta reconstruction requires all preceding values. A delta SubIntSplit stream
is therefore read sequentially from row zero: `skip()` decodes every skipped
row, `readWithVisitor()` reads through `materialize()`, and point or range reads
cost a full scan. Keep this option disabled for production until the format has
restart points.

### Row frame

Some packed IDs carry a per-row counter: XMark pre/post identifiers grow by
2^28 + 1 per row in their low 56 bits, and the high half of a UUIDv7 steps by
one within each millisecond. No bit boundary isolates such a counter, because
its carries cross whichever boundary the planner picks. With
`TuningConfig::rowFrame`, on by default, the writer subtracts a predictor
`slope * row + base` from every value, in the physical type's modular
arithmetic, and plans the sections over the residuals. `RowFrame.h` fits two
kinds of frame:

- A line frame. `fitRowFrame` reads the growth of the low bits over strides of
  1,024 rows and takes the widest low-bit width at which at least 90% of the
  strides agree with the median slope to within half a stride. The slope must
  also hold over strides of 1,000 rows, which rules out a field whose period
  divides the first stride. A column needs at least 16 strides. The base is
  the most negative residual of those low bits, so the fields below the width
  stay non-negative and do not borrow from the fields above them.
- A step frame, tried only when no line fits. `fitStepFrame` takes the most
  common non-zero step between adjacent rows, found with three Misra-Gries
  counters and then counted exactly, when at least a quarter of the row pairs
  take it. Its base is zero.

A fitted frame is kept only where it encodes smaller. A line frame is priced
by the split DP over the residuals and over the values, the frame block
included, and the winner's cost grid is the one the encoder plans on. A step
frame turns counters into runs of whole values that the planner's run models
misprice, so the writer encodes both forms and keeps the smaller. A
preserve-mode replay takes a frame exactly when the captured layout recorded
one. The delta form never carries a frame. `TuningConfig::rowFrameForceApply`
keeps any fitted frame without the comparison and exists for ablation only.

Reads add the frame back after the sections are reassembled, one add per row,
from the row's position in the stream. `skip()` only moves the cursor, so a
framed stream keeps random access. The visitor fast path already decodes
through `materialize()`, and the slow path defers to `materialize()` for a
framed stream. `SubIntSplitEncodingView` adds the frame back on every read
path, so a framed stream keeps positional random access through the view.

### Section transforms

A section that compresses poorly in row order can compress well once its rows
are sorted by another section's value, since sorting groups like values for RLE
and MainlyConstant children. `SectionTransform.h` defines one such transform,
the key-derived permutation (transform id 1). One section is the key and is
stored in row order. Every section that takes the transform is stably sorted by
the key's value before it is encoded. No permutation is stored: a reader
decodes the key, rebuilds the same stable order with a radix sort, and gathers
each permuted section back to row order while it assembles the values.

The writer applies it only on request:

- `TuningConfig::transform` names the transform, and `TuningConfig::autoTransform`
  asks the encoder to consider the key-derived permutation on its own. Both
  are off by default, so default streams are unchanged.
- `TuningConfig::keySection` pins the key. At its default, 0xFF, the encoder
  prices one attempt per candidate key, bounded by the best attempt so far,
  and keeps the smallest. A pin past the stream's last section is searched the
  same way, since the split depends on the data.
- A key is refused when it has more than a quarter as many distinct values as
  rows, since sorting by it would group almost nothing.
- Each section keeps the transform only where its encoding, plus an 8-byte
  margin, is strictly smaller than the section in row order.
- `TuningConfig::forceApply` applies the transform to every eligible section
  whatever it costs, still searching the key when none is pinned. It exists
  for ablation only.

A transform is undone over the whole column, so a reordered stream decodes the
column once and serves partial reads from a cache. `skip()` only moves the
cursor. The visitor slow path defers to `materialize()`. `SubIntSplitEncodingView`
reads reordered streams positionally; see
[Random access through the view](#random-access-through-the-view).

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

SubIntSplit is not in `defaultEncodingReadFactors()`, so default selection
never chooses it. A writer opts in by adding SubIntSplit to its read factors;
selection then prices it with the estimate below and can screen it with the
admission gate. Making it a default waits for reader coverage of every
encoding a section can take: `EncodingView` support for Varint, Delta and
FrequencyPartition, and a SubIntSplit case in the legacy encoding factory.
Explicit serde routing works as before.

Explicit routing may produce a one-section encoding when the configured stream
has no profitable split. Benchmark each routed stream because the wrapper adds
metadata without decomposing the value in that case.

### Admission into default selection

`SubIntSplitEncoding::estimateSize()` runs the same split DP as the encoder
over a smaller sample (512 values in 64-row blocks), on size alone, and adds
the header bytes of the plan it finds, under the production `TuningConfig`.
With `TuningConfig::rowFrame` it also plans the sample less a row frame fitted
to the whole column, and keeps the smaller. The answer is floored at
FixedBitWidth's exact estimate, which the whole-value floor guarantees a
written stream never exceeds, and is that floor when no plan can be priced. It
returns nothing, so selection skips the candidate, when the policy's streams
go to a substream compressor and `subIntSplit.estimateCompressionGuard` is set,
since the estimate counts uncompressed bytes. With
`subIntSplit.estimateBitFlipScreen`, `estimateSizeLowerBound()` answers the
floor without planning where the bit-flip gradient gate finds no field
boundary, which lets selection skip the DP for a stream that cannot win.

`subIntSplit.admission` can put a cheap screen in front of the estimate. The
screen is a bit-flip profile: per bit position, how often that bit differs
between consecutive values, over `subIntSplit.admissionProfilePairs` strided
pairs. `kBitFlip` offers SubIntSplit as a candidate only when the profile's
gradient shows a field boundary (`bitFlipGradientGate()`); `kBitFlipEntropy`
also requires the flipping bits to be less than random. An admitted stream
still has to win on size unless `subIntSplit.admissionForces` is set, which
selects an admitted stream without pricing it.
`subIntSplit.inNestedStreams = false` keeps nested streams such as RLE run
values from choosing SubIntSplit.

## Tuning options

`subintsplit::kDefaultTuningConfig` owns the algorithm's production defaults.
They are internal to SubIntSplit instead of fields on the generic
`Encoding::Options` shared by every encoding.

| Setting | Default | Effect |
|---|---:|---|
| `sampler.maxSamples` | 2,048 | Bound values sampled by the planner |
| `selector.boundaryPruneThreshold` | `0.001` | Discard weak adjacent bit-plane changes |
| `selector.maxCandidateBoundaries` | Unlimited | Bound the split grid when configured by a benchmark |
| `selector.maxSectionWidth` | Unlimited | Skip wide grid cells except the full-range fallback |
| `selector.frequencyMetricsMaxWidth` | Unlimited | Skip frequency metrics above a width |
| `selector.decodeCostBitsPerValue` | `0.0` bits/value | Penalize each extra dynamic section |
| `selector.allowHuffman`, `selector.allowDeltaBlock` | `false` | Let the planner price Huffman or DeltaBlock |
| `selector.trimConstantPlanes` | `true` | Trim constant edge planes before the DP |
| `allowedEncodings` | empty | Restrict the encodings the planner prices; empty allows all |
| `decodeChunkSize` | 4,096 values | Bound values reconstructed per decode pass |
| `foldConstantSections` | `true` | Fold Constant sections into one word at open |
| `passThrough` | `true` | Decode a sole verbatim section into the caller's buffer |
| `visitorBlockBuffer` | `true` | Decode the visitor slow path a block at a time |
| `rowFrame` | `true` | Fit a row frame and keep it where it encodes smaller |
| `rowFrameForceApply` | `false` | Ablation only: keep any fitted row frame unpriced |
| `transform` | `0` | Section transform to offer; 1 is the key-derived permutation |
| `autoTransform` | `false` | Offer the key-derived permutation per section |
| `keySection` | `0xFF` | Key section; 0xFF searches every section |
| `forceApply` | `false` | Ablation only: apply the transform to every eligible section |
| `selector.decodeWeighting` | weight `0.0`, `Bulk`, `Cursor` | Decode cost charged to sections, and the read shape and reader it is priced for |
| `maxSizeRegression` | `0.05` | Size a decode-weighted choice may give up |
| `hybridPlanner` | `false` | Re-price a DP shortlist and refine the winner |
| `sectionExecutor` | null | Encode sections concurrently |

Benchmarks and focused tests can pass an alternate `TuningConfig` directly to
`SubIntSplitEncoding`. Normal writer and reader paths always use the constant
default. Planner changes can alter section boundaries and encoded bytes; the
decode settings do not alter the format.

`Encoding::Options::subIntSplitDeltaPreTransform` remains separate because it
is a runtime rollout gate that changes the stream representation.
`Encoding::Options::subIntSplit` is not a setting either: it carries the
decode weighting SubIntSplit gives its sections' own encoding selection, and
callers leave it unset.

Top-level selection's SubIntSplit settings stay on
`Encoding::Options::subIntSplit` (`subintsplit::Options`), since selection sees
only those options:

| Setting | Default | Effect |
|---|---:|---|
| `subIntSplit.admission` | `kEstimate` | Screen candidacy with the bit-flip profile |
| `subIntSplit.admissionForces` | `false` | Let an admitted stream skip the size comparison |
| `subIntSplit.admissionProfilePairs` | `1024` | Pairs the admission profile samples; zero is every pair |
| `subIntSplit.inNestedStreams` | `true` | Let nested streams choose SubIntSplit |
| `subIntSplit.estimateCompressionGuard` | `true` | Decline to estimate under substream compression |
| `subIntSplit.estimateBitFlipScreen` | `false` | Bound the estimate from the bit-flip gradient gate before planning |

Treat these as benchmark controls rather than table-level contracts. Changing
planner options can change section boundaries and encoded bytes. Reader-only
chunk size changes do not alter the format.

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
| `BitFlipProfile.h` | Admission statistics | Integral values; produces per-bit flip probabilities, their variance and gradient, and optionally the varying bits | Sampled pairs keep adjacency; the scalar and AVX2 counters agree |
| `BitSection.h` | Shared planner/encoder vocabulary | Inclusive first and last bits; produces section ranges and `SectionPlan` entries | Ranges use physical bit positions, remain ordered least-significant first, and report exact widths |
| `CostModel.h` | Planner cost models | Section metrics, width, and value count; exposes candidate costs and pruning decisions | Each model mirrors selection's size estimator for its encoding; costs use bits consistently and impossible candidates remain distinguishable from expensive ones |
| `DecodeCost.h` | Planner decode-cost weighting | Section encodings and access pattern; produces per-section read costs | Zero weight reproduces size-only planning exactly |
| `DeltaTransform.h` | Optional write transform and sequential read recovery | Physical values ↔ first value plus zigzag residuals | Arithmetic is bit-preserving for signed extrema; transformed streams require decoding from row zero |
| `Estimator.h` | Benchmark and test helper | Values and gate config; returns the gate decision and an ungated DP cost | Not used by production selection |
| `Format.h` | Persistent write/read boundary | Section count, flags, ranges, child sizes, and payload offsets; `parseSections()` validates them | Header sizes and field order remain compatible with stored data; malformed headers fail with `NIMBLE_CHECK_FILE` |
| `TuningConfig.h` | Planner and decoder tuning | `TuningConfig`, `kDefaultTuningConfig` | Defaults preserve standard behavior |
| `RowFrame.h` | Optional write transform and per-row read recovery | Physical values ↔ residuals from a fitted `slope * row + base` | Arithmetic is modular in the physical width; a read adds the frame back from the row index alone |
| `Options.h` | Section selection plumbing | `subintsplit::Options`, held by `Encoding::Options::subIntSplit` and set by `sectionEncodingOptions` | Callers leave it unset |
| `PlanRefiner.h`, `PlanRefiner.cpp` | Hybrid planner | Planner sample, DP shortlist, and a section's options; returns the refined plan and its estimated size | Prices ranges as `ManualEncodingSelectionPolicy::select` would; kept in step with it by hand |
| `Sampler.h` | Planner-only preprocessing | Full physical-value span and `SamplerConfig`; produces `uint64_t` samples | Sampling is bounded, deterministic, and block-stratified so local runs survive |
| `SectionAccumulator.h` | Full and selective decode hot path | Decoded unsigned section values, range masks, shifts, and an output span | Scalar and AVX2 paths produce identical physical bits and never leak bits outside a section width |
| `SectionMetrics.h` | Planner statistics | Candidate-range samples and reusable frequency storage; produces the range, run, cardinality, and dominant-value measurements the cost models read | Metrics describe extracted section values; exact distinct counts stop at the configured cap and scratch state is reset between ranges |
| `SectionTransform.h` | Optional write transform and read recovery | Declares transforms, `TransformId`, and the key-order helpers | Transform ids are persisted and never renumbered; unknown ids fail as file errors |
| `SectionTransforms.cpp` | Key-derived permutation | Key section values ↔ stable sort order and run bookkeeping | The order is a stable sort, rebuilt identically by writer and reader |
| `SplitBoundaries.h` | Preserved-layout configuration interface | Declares config keys and parse/serialize APIs for ranges and candidate exclusions | Preserve mode describes a complete physical-width partition |
| `SplitBoundaries.cpp` | Preserved-layout parsing and validation | Text configs ↔ section plans | Rejects gaps, overlaps, malformed endpoints, out-of-range bits, and exclusion-count mismatches |
| `SplitSelector.h` | Planner entry point and dynamic program | Samples, physical width, full row count, and `SelectorConfig`; returns `SelectorResult` | Sentinel defaults preserve standard planning, reported total cost matches the returned plan, and output covers the full width |
| `SplitSelector.cpp` | Boundary search | Active-bit statistics; produces retained boundaries and the grid layout | Both active-range edges remain candidates and configured caps remain hard limits |
| `TopLevelPolicy.h` | Admission gates | `BitFlipProfile` and `TopLevelPolicyConfig`; decides whether SubIntSplit is a candidate | `kEstimate` always admits; gates only decide candidacy unless forcing is asked for |

### Registration, selection, and writer configuration

| File | Responsibility and reason to change it |
|---|---|
| `encodings/SubIntSplitEncoding.h` | Owns the public encoding implementation. It coordinates planning or layout replay, optional delta comparison, child encoding, serialization, child decoder construction and cursors, full materialization, skipping, reset, and visitor reads. Start here when behavior spans more than one core helper. |
| `encodings/common/Encoding.h` | Declares `subIntSplitDeltaPreTransform`, SubIntSplit's one runtime option; tuning lives in `subintsplit/TuningConfig.h`. |
| `encodings/selection/EncodingSelectionPolicy.h` | `nestedEncodingReadFactors()` decides the candidates a section is selected from. Change it together with the planner's cost models. |
| `encodings/common/EncodingFactory.cpp` | Constructs typed SubIntSplit decoders on the normal factory path. |
| `encodings/legacy/EncodingFactory.cpp` | Constructs the same physical types through the legacy visitor dispatch path. |
| `encodings/views/SubIntSplitEncodingView.h` | Implements random access over supported nested section encodings, including row frames and key-derived reordering. Change it when adding a view-capable child or adjusting per-row reconstruction. |
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
| `encodings/tests/SubIntSplitCostModelsTest.cpp` | Per-encoding cost models and metric collection. |
| `encodings/tests/SubIntSplitSelectorTest.cpp` | Range and partition counts, decode-cost weighting, and plan ranking. |
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
| Wire format or flags | `subintsplit/Format.h` and `SubIntSplitEncoding::writeEncoding()` | `parseSections()`, legacy factory, and old-format compatibility | Encoding and configured-split tests |
| Boundary or partition heuristic | `SplitSelector.*` | `Sampler.h`, `SectionMetrics.h`, and `CostModel.h` | Planner-option and section-metrics tests; optimized encode benchmark |
| Nested child choice | `SubIntSplitEncoding::encodeResiduals()` | `CostModel.h`, `nestedEncodingReadFactors()`, and view support | Section-candidate, encoding-view, and production-corpus tests |
| Full-decode performance | `SubIntSplitEncoding::materialize()` | `SectionAccumulator.h` and the decoder constructor | Encoding fuzzer and full-decode benchmarks |
| Filter or projection performance | `SubIntSplitEncoding::readWithVisitor()` | `bulkScan()`, pending-block state, and `SubIntSplitEncodingView.h` | View fuzzer and selective-reader benchmark |
| Serde property behavior | `velox/NimbleConfig.*` | option builder, `WriterOptions.h`, and `Writer.cpp` | Writer and option-builder tests |
| Selection eligibility | `EncodingSelectionPolicy.cpp` | Production factors and explicit layouts | Production-corpus comparison against the current default selector |
| Delta transform | `DeltaTransform.h` | format flags and sequential reader restrictions | Delta tests plus full sequential round trips |
