# ALP_RD encoding

ALP_RD preserves the IEEE 754 bit pattern of each `float` or `double`. It splits
the unsigned representation into a high part and a low part, replaces common
high parts with small dictionary codes, and stores dictionary misses separately.
It performs no floating-point arithmetic. Signed zero, infinity, subnormal
values, and NaN payloads are preserved.

ALP_RD is a separate encoding from ALP. ALP uses a decimal transformation into
integers; ALP_RD encodes the original binary representation. An ALP_RD payload
does not contain ALP blocks or switch between the two algorithms internally.

## Scope and selection

ALP_RD supports explicit layouts and opt-in manual selection, ordinary Nimble
file reads, and the generic visitor path. It has no encoding view yet.

To enable automatic selection, add `ALPRD=<readFactor>` to the manual policy's
candidate configuration. ALP_RD is absent from production default candidates.
ALP and ALP_RD compete with all eligible encodings at the same selection node
using `estimatedSize * readFactor`; there is no dedicated ALP-failure fallback
or special mutual exclusion. A lower factor favors an encoding. A size win
alone does not establish a decoding-speed win.

The encoding applies to one Nimble encoding payload. Each payload has its own
split width, dictionary, child encodings, and exception list. There is no fixed
1,024-row block size or sharing of dictionaries across stream chunks. Row counts
and exception positions use Nimble's 32-bit row range. Empty input is not
representable by ALP_RD; the surrounding writer can use its ordinary fallback.

Children use Nimble's existing nested selection and layout replay mechanisms.
Their types are integers, so ALP and ALP_RD cannot directly encode these children.
This is a type constraint, not a global exclusion between ALP and ALP_RD in a
larger encoding tree.

Dictionary alphabets, RLE run values, and MainlyConstant uncommon values can
select ALP_RD through their inherited or explicitly overridden candidates.
Parent estimation includes that floating-point child choice. Ancestor encoding
filters apply as usual.

A Nullable wrapper retains physical selection and type tags by default. When
its policy enables ALP_RD, the data child uses logical floating-point selection
so ALP_RD and floating-point containers remain eligible. Containers containing
ALP_RD retain their logical type during layout replay. Leaf encodings other
than ALP/ALP_RD keep their physical tags. Explicit ALP/ALP_RD layouts also retain
the logical type. The default nullable layout is unchanged.

## Binary layout

The encoding ID is **26** (`EncodingType::ALPRD`). Only the `Float` and `Double`
logical data types are valid. Let `N` be the row count and `W` the value width in
bits, either 32 or 64.

All fixed-width integers are little-endian. `varint32` denotes an unsigned
base-128 variable-length integer with at most five bytes. Each byte contributes
seven low-order bits; bit 7 indicates another byte. The fifth byte must be less
than 16.

| Field | Representation | Meaning |
| --- | --- | --- |
| Encoding type | `uint8` | 26 |
| Data type | `uint8` | Nimble `Float` or `Double` type ID |
| Row count (`N`) | `uint32` or `varint32` | Common Nimble prefix; `N > 0` |
| Right bit width (`R`) | `uint8` | `W - 16 <= R < W` |
| Dictionary size (`D`) | `uint8` | `1 <= D <= 8` |
| Exception count (`E`) | `varint32` | Always present; `0 <= E <= N` |
| High-part dictionary | `uint16[D]` | Distinct values, each less than `2^(W-R)` |
| Codes child length | `varint32` | Byte length of the complete child encoding |
| Codes child | bytes | Non-null `Uint16`, `N` values |
| Right-parts child length | `varint32` | Byte length of the complete child encoding |
| Right-parts child | bytes | Non-null `Uint32` for FLOAT or `Uint64` for DOUBLE, `N` values |
| Exception-positions child length | `varint32` | Present only when `E > 0` |
| Exception-positions child | bytes | Non-null `Uint32`, `E` values; present only when `E > 0` |
| Exception-high-parts child length | `varint32` | Present only when `E > 0` |
| Exception-high-parts child | bytes | Non-null `Uint16`, `E` values; present only when `E > 0` |

The common prefix follows `Encoding::Options::useVarintRowCount`. The same option
applies recursively to all children. It is supplied by the containing stream;
there is no additional prefix-format flag inside ALP_RD. All other variable
lengths and counts shown as `varint32` always use that representation.

There is no outer compression byte. Every child includes its own common prefix
and may use a supported non-null integer encoding and its associated compression.
Its declared type and row count must match the table. The payload ends after
the last child, without additional bytes.

### Reconstruction and exceptions

For each row `i`, require `codes[i] < D` and `rightParts[i] < 2^R`. Recover the
unsigned value bits as:

```text
bits[i] = (dictionary[codes[i]] << R) | rightParts[i]
```

Exception positions are zero-based positions in this payload. They must be
strictly increasing and less than `N`. For every exception `j`, its high part
must be less than `2^(W-R)`. Replace the high part at the indicated position:

```text
i = exceptionPositions[j]
bits[i] = (exceptionHighParts[j] << R) | rightParts[i]
```

The code at an exception position must still be a valid dictionary index. The
writer emits code zero there. The right-parts child retains the original low
bits for every row, including exceptions. Finally, reinterpret `bits[i]` as the
logical floating-point type without numeric conversion.

For example, a DOUBLE payload with `R = 48`, dictionary `[0x3ff0]`, codes
`[0, 0]`, right parts `[1, 2]`, and one exception `(position = 1, high = 0xc000)`
recovers the bit patterns `0x3ff0000000000001` and `0xc000000000000002`.

### NULL handling

ALP_RD does not store NULLs. Nimble's `Nullable` wrapper supplies the null stream
and passes only the non-null values to ALP_RD, retaining the FLOAT/DOUBLE logical
type. `N` and exception positions therefore refer to the non-null value stream,
not the original nullable row positions. An all-null chunk does not require an
empty ALP_RD payload.

## Writer parameters and layout replay

The writer samples at most 1,024 values from evenly sized intervals, varying
the offset within each interval deterministically to avoid aliasing periodic
input. It evaluates high-part widths from 1 through 16 with dictionaries of one
through eight frequent prefixes. A cheap scalar model includes Constant,
Trivial and byte-rounded or exact-bit FixedBitWidth costs. Equivalent scalar
layouts are grouped so they do not crowd out other split shapes. At most four
candidate splits are then scored
using the actual child selection policies, including their candidate filters,
read factors, and explicit layout bindings. No candidate payload is encoded.

The final score includes the ALP_RD prefix, dictionary entries, exception count,
child-length varints and selected child sizes. Scalar sizes include prefix
options and FixedBitWidth's seven padding bytes. Estimation and encoding share
this training routine. Equal final costs prefer narrower low parts; equal
scalar layouts within a split prefer the smaller dictionary.

Sampling, the bounded shortlist and existing composite child estimates remain
heuristics. Scalar estimates project observed ranges and frequencies to the
full row count; composite estimates project sampled sizes. Floating-point
container estimates sample their derived value stream and retain the existing
heuristics for integer or boolean sibling streams. The manual policy uses one
level of child-policy lookahead: sampled child selection uses existing container
heuristics instead of recursively training every possible encoding tree. The
writer selects again on the actual child input at each level. Generic compression
is not predicted, matching Nimble's existing in-memory selection objective.
Neither training nor automatic selection guarantees the smallest serialized payload.
The full input is encoded against the selected dictionary; unsampled keys
become exceptions. Readers depend only on the serialized parameters and
reconstruction rules, so training may evolve without changing the format.

Layout capture records the ALP_RD ID and four child slots, with absent layouts
for the two exception children when there are no exceptions. These slots let
the replay policy select layouts for newly appearing exception streams. Replay
recomputes the dictionary and split from the new values. It does not reuse the
previous payload's dictionary. Newly appearing exceptions use the replay
policy's normal child fallback. Selection remains greedy: an explicit or
replayed ALP_RD layout is honored even when another encoding would be smaller;
the writer does not encode the full input and then re-encode it for comparison.

A fixed layout can specify all four children, for example:

```cpp
const EncodingLayout child{
    EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
const EncodingLayout layout{
    EncodingType::ALPRD,
    {},
    CompressionType::Uncompressed,
    {child, child, child, child}};
```

Use this layout with `ReplayedEncodingSelectionPolicy<float/double>` or the
appropriate scalar stream in `WriterOptions::encodingLayoutTree`.

## Reader behavior

The native and legacy reader paths share the same ALP_RD decoder.

The basic decoder keeps independent cursors for the codes and right-parts
children. It eagerly decodes exception positions and high parts, validates their
bounds/order, and patches matching rows during materialization. It does not
materialize either complete main child on behalf of ALP_RD; each child codec
retains its own decoding strategy.

After metadata validation, both exception arrays are allocated for the full
declared exception count through the caller's MemoryPool. Each exception child
is materialized in one call, then all exception values are validated before
the decoder is returned or slicing uses the positions. Exceeding a configured
pool limit can therefore fail before invalid exception values are detected.
There is no count-to-compressed-size limit: valid child encodings can represent
more exceptions than their payload has bytes. Loading remains eager and uses
O(exceptionCount) memory, including for selective reads. Child codecs retain
their own allocation behavior.

`skip` advances both main children and the exception cursor. `reset` restores
them to the beginning. The generic visitor uses these same operations for
correct filtering and selected-row reads. Optimized selective reading and
encoding views can be added independently while retaining this format.

Slicing preserves the dictionary and split, slices both main children, keeps
only exceptions in the requested range, and rebases their positions to zero.
It uses the shared exception loader to validate both exception streams without
constructing a complete ALP_RD decoder. It omits both exception children when
the slice has no exceptions. A slice must contain at least one row and stay
within the original row range.

## Validation and benchmark

`ALPRDEncodingTest` covers independent wire fixtures, bit-exact round trips,
special IEEE values, fixed and varint prefixes, exceptions, cursor operations,
slices, layout replay, malformed metadata, split-cost ties, and preservation of
existing nullable physical child layouts. It separately checks memory-pool
limits during exception loading, invalid positions throughout the exception
stream, and valid compressed exception streams whose counts exceed their
payload sizes.
`ALPRDSelectionTest` checks exact leaf sizes, bounded sampling, positive and
negative choices, read factors, candidate inheritance, floating-point nesting,
nullable layouts and replay. Random selection and the writer fuzzer also offer
ALP_RD, including shared-prefix workloads and unmodified special values.
`FloatingPointColumnReaderTest` and
`ALPRDColumnReaderTest` verify real file reads, filters, NULLs, multiple chunks,
and layout caching through both the native and legacy reader paths.
`ReadWithVisitorTest` adds sparse input rows, selected and skipped exceptions,
special IEEE bit patterns, NULL mapping, filtering, and filter-only reads across
batches and chunks. `StreamSlicerTest` verifies exception rebasing for raw
serialized streams and uncompressed or compressed tablet chunks.

The `nimble_alprd_benchmark` target provides FLOAT/DOUBLE workloads with shared
high parts, multiple prefixes and exceptions, plus broad-exponent and random-bit
controls. It also accepts external IEEE binary data, including the ALP paper's
POI-lat and POI-lon samples. The `--profile` mode reports encoded bits/value,
exception counts and repeated wall/CPU timings for training, complete encoding,
construction-plus-decoding, and decoding with a reused decoder.

Data generation/loading, encoded snapshot copying and caller output allocation
are outside timing. Complete encoding includes training and child selection;
construction-plus-decoding includes exception loading and destruction. Both
decoding lifecycles are validated bit-for-bit before timing.

The `--selection_profile` mode compares a configured baseline candidate set
against the same set with ALP_RD added. It reports serialized trees, estimated
and actual bytes, and separate selection, encoding and construction-plus-decoding
wall/CPU times. The default benchmark configuration uses equal read factors for
size comparisons; `--selection_read_factors` and `--alprd_read_factor` make the
weights explicit. These are encoding-payload measurements, not file-scan
measurements. Selection timing creates fresh lazy statistics on each iteration.
