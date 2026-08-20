# Nimble Encodings Documentation


This document describes Nimble's encoding system, including the available encoding
types, the cascading (recursive) and encoding selection policies.

## Encoding Types

Nimble supports various encoding types which are extensible.
The following table shows the list of already supported encodings.


| Encoding | Data Sequence | Description |
|----------|-------------|-------------|
| **Trivial** | Any | It is baseline encoding. Raw values with no transformation. |
| **Constant** | All identical | Stores a single value for the entire block. |
| **MainlyConstant** | Mostly identical | Stores a common value plus exceptions. |
| **RLE** | Repeated runs | Run-length encoding: (value, count) pairs. |
| **Dictionary** | Low cardinality | Alphabet of unique values + index array. |
| **FixedBitWidth** | Bounded integers | Bit-packs integers using the minimum bits needed. |
| **Varint** | Variable integers | Variable-length integer encoding. |
| **Delta** | Sequential/trending | Stores deltas between consecutive values. |
| **Sentinel** | Nullable data | It uses a sentinel value to represent nulls. |
| **Nullable** | Nullable data | Explicit null bitmap + non-null values stream. |
| **SparseBool** | Skewed on false/true | Stores positions of the minority boolean value. |
| **Prefix** | Shared prefixes | Prefix-based string encoding. |

## Cascading (Recursive) Encoding

Nimble's key differentiator is **cascading encoding**: encodings can contain other
encodings, and helps forming an encoding tree.
*  Each nested encoding is independently selected based on the characteristics of its sub-stream data.
* When selecting an encoding for a nested sub-stream, the parent encoding type is
excluded from the candidate list to prevent infinite recursion.
* For example, a `DictionaryEncoding`'s alphabet sub-stream will not be encoded with another `DictionaryEncoding`.

### Example: Dictionary Encoding Tree

For an example, consider an input array `["apple", "banana", "apple", "cherry", "banana", "strawberry"]`:

```
DictionaryEncoding<string>
├── Alphabet: TrivialEncoding<string>
│   └── ["apple", "banana", "cherry",  "strawberry"]
│       └── Compression: Zstd (if beneficial)
└── Indices: FixedBitWidthEncoding<uint32_t>
    └── [0, 1, 0, 2, 1, 3] (2 bits each)
        └── Compression: Zstd (if beneficial)
```

The dictionary encoding produces two sub-streams:
1. **Alphabet**: The unique values, encoded with whatever encoding best fits them.
2. **Indices**: Integer references into the alphabet, typically bit-packed.

Each sub-stream is recursively encoded using the same selection process, and
compression is independently applied at each level.

## Encoding Selection

Encoding selection is controlled by pluggable **EncodingSelectionPolicy** implementations.
The policy inspects the data and its statistics, then selects the best encoding.

### Selection Flow

1. Compute `Statistics<T>` for the data (unique counts, min/max, value distribution)
2. Call `policy->select(values, statistics)` to get an `EncodingSelectionResult`
3. Create an `EncodingSelection` context with the result, statistics, and policy
4. Dispatch to the selected encoding's `encode()` method
5. If the encoding is nested, `encodeNested()` triggers recursive selection

### ManualEncodingSelectionPolicy (Default)

Uses **cost-based selection**: evaluates each candidate encoding, estimates the encoded
size using `EncodingSizeEstimation`, and applies a read performance factor:

```
cost = estimatedSize × readFactor
```

Read factors weight encodings by decode performance:

| Encoding | Read Factor | Rationale |
|----------|------------|-----------|
| Trivial | 0.7 | Fast decode + compression benefits |
| FixedBitWidth | 0.9 | Faster than variable-width |
| Dictionary | 1.0 | Standard |
| RLE | 1.0 | Standard |
| MainlyConstant | 1.0 | Standard |

The encoding with the lowest cost wins.

### LearnedEncodingSelectionPolicy

Uses an **ML model** to predict the best encoding choice based on data statistics.
Intended for workloads where the cost model doesn't capture the full picture

### ReplayedEncodingSelectionPolicy

Replays a previously captured **EncodingLayout** to reproduce an exact encoding tree.
Useful for:
- Testing: ensuring deterministic encoding across runs
- Debugging: reproducing a specific encoding configuration
- Migration: applying a known-good layout to new data

## Compression

Compression is applied **per encoding level**, after the encoding produces its byte
output. The compression policy decides whether to keep the compressed version based
on the compression ratio.


## Statistics

The `Statistics<T>` class computes per-block statistics used by encoding selection:
- Unique value count (NDV)
- Min/max repeat lengths (for RLE)
- Value distribution characteristics
- Min/max values
- BucketCounts
Column-level statistics (across stripes) are computed by the `velox/stats/` module
for use in query planning and data skipping.
