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

The initial implementation supports explicit encoding and fixed/replayed layouts,
including ordinary Nimble file reads and the generic visitor path. It does not
add ALP_RD to automatic encoding selection or provide an encoding view.

The encoding applies to one Nimble encoding payload. Each payload has its own
split width, dictionary, child encodings, and exception list. There is no fixed
1,024-row block size or sharing of dictionaries across stream chunks. Row counts
and exception positions use Nimble's 32-bit row range. Empty input is not
representable by ALP_RD; the surrounding writer can use its ordinary fallback.

Children use Nimble's existing nested selection and layout replay mechanisms.
Their types are integers, so ALP and ALP_RD cannot directly encode these children.
This is a type constraint, not a global exclusion between ALP and ALP_RD in a
larger encoding tree.

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

The current writer samples at most 1,024 evenly spaced values and evaluates
high-part widths from 1 through 16. For each width it builds a dictionary from
the eight most frequent sampled high parts, or all keys when there are fewer
than eight. It estimates packed codes, right parts, dictionary entries, and
32-bit-position/16-bit-high exceptions to choose the split. The full input is
then encoded against that dictionary; unsampled keys become exceptions.

This is an internal parameter heuristic, not an automatic encoding-selection
cost model or a requirement for other writers. Readers depend only on the
serialized parameters and reconstruction rules. Future writers may improve
training without changing this layout.

Layout capture records the ALP_RD ID and four child slots, with absent layouts
for the two exception children when there are no exceptions. These slots let
the replay policy select layouts for newly appearing exception streams. Replay
recomputes the dictionary and split from the new values. It does not reuse the
previous payload's dictionary. Newly appearing exceptions use the replay
policy's normal child fallback.

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

`skip` advances both main children and the exception cursor. `reset` restores
them to the beginning. The generic visitor uses these same operations for
correct filtering and selected-row reads. Optimized selective reading and
encoding views can be added independently while retaining this format.

Slicing preserves the dictionary and split, slices both main children, keeps
only exceptions in the requested range, and rebases their positions to zero.
It omits both exception children when the slice has no exceptions. A slice must
contain at least one row and stay within the original row range.

## Validation and benchmark

`ALPRDEncodingTest` covers independent wire fixtures, bit-exact round trips,
special IEEE values, fixed and varint prefixes, exceptions, cursor operations,
slices, layout replay, and malformed metadata. `FloatingPointColumnReaderTest`
verifies real file reads, filters, NULLs, multiple chunks, and layout caching
through both the native and legacy reader paths.

The `nimble_alprd_benchmark` target provides FLOAT/DOUBLE encode and
construction-plus-bulk-decode cases over 100,000 values, with common high parts
and with a one-percent tail of uncommon high parts. Data generation is outside
the measured interval. Encoding includes training and nested encoding selection;
decoding includes construction, exception materialization, and destruction, but
excludes allocation of the caller's output array. The benchmark validates the
decoded bit patterns and the presence/absence of exceptions before timing.
