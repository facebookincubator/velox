# Adding a New Encoding to the Nimble Encoding Module

This guide walks through every step required to add a new encoding to the Nimble encoding module, from defining the type through enabling it in the encoding selection pipeline.

## Architecture Overview

Nimble uses **cascading (recursive) encodings** where encodings can wrap other encodings. The write path selects encodings via an `EncodingSelectionPolicy`, serializes data through the static `encode()` method on each encoding class, and the read path deserializes via `EncodingFactory::create()`.

```
Write Path:
  Data → EncodingSelectionPolicy::select() → EncodingType
       → EncodingFactory::encode<T>() → MyEncoding<T>::encode()
       → serialized bytes (with nested encodings for sub-streams)

Read Path:
  Serialized bytes → EncodingFactory::create()
       → reads EncodingType from prefix → switch → MyEncoding<T>(data)
       → skip() / materialize() / readWithVisitor()
```

The key directories:

| Directory | Contents |
|-----------|----------|
| `encodings/common/` | Base classes (`Encoding.h`), factory (`EncodingFactory.h/.cpp`), primitives, layout |
| `encodings/selection/` | Encoding selection policy, size estimation, statistics |
| `encodings/` | Concrete encoding implementations |
| `encodings/tests/` | Unit tests and test utilities |

## Step-by-Step Guide

### Step 1: Add the `EncodingType` Enum Value

**File:** `velox/dwio/nimble/common/Types.h`

Append a new value to the `EncodingType` enum:

```cpp
enum class EncodingType {
  Trivial = 0,
  RLE = 1,
  // ... existing values ...
  Prefix = 11,
  MyEncoding = 12,  // <-- add here
};
```

**File:** `velox/dwio/nimble/common/Types.cpp`

Add a case to the `toString(EncodingType)` switch:

```cpp
CASE(MyEncoding);
```

### Step 2: Implement the Encoding

Create `velox/dwio/nimble/encodings/MyEncoding.h` (and optionally `.cpp` for non-template code).

Every encoding must inherit from `TypedEncoding<T, physicalType>` (defined in `encodings/common/Encoding.h`) and implement:

#### Required Methods

```cpp
#pragma once

#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

namespace facebook::nimble {

template <typename T>
class MyEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  // Constructor: deserialize from binary data.
  // Called by EncodingFactory::create().
  MyEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  // Reset internal read state to the beginning.
  void reset() final;

  // Skip rowCount rows in the read stream.
  void skip(uint32_t rowCount) final;

  // Materialize rowCount rows into the output buffer.
  void materialize(uint32_t rowCount, void* buffer) final;

  // Template method for vectorized reading (selective reader path).
  // Not virtual — dispatched via EncodingUtils.h.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  // Static method: encode data into binary format.
  // Called by EncodingFactory::encode<T>().
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});
};

} // namespace facebook::nimble
```

#### Data Layout

Every encoding starts with a standard prefix (serialized by `Encoding::serializePrefix()`):

| Bytes | Content |
|-------|---------|
| 1 | `EncodingType` |
| 1 | `DataType` |
| 4 (or varint) | Row count |

After the prefix, the encoding writes its own data. Access the start of encoding-specific data via `this->dataOffset()` in the constructor.

#### Encoding Sub-streams (Nested Encodings)

If your encoding has nested sub-streams (e.g., Dictionary has alphabet + indices), use `EncodingSelection::encodeNested<T>()` to recursively encode them:

```cpp
// In encode():
std::string_view serializedChild =
    selection.template encodeNested<uint32_t>(
        EncodingIdentifiers::MyEncoding::ChildStream,
        childData,
        tempBuffer,
        options);
```

And define identifiers in Step 3.

#### Supported Data Types

Choose which data types your encoding supports. The factory macros determine dispatch:

| Macro | Types |
|-------|-------|
| `RETURN_ENCODING_BY_LEAF_TYPE` | All types including bool and string |
| `RETURN_ENCODING_BY_NON_BOOL_TYPE` | All except bool |
| `RETURN_ENCODING_BY_NUMERIC_TYPE` | Numeric types only (no bool, no string) |
| `RETURN_ENCODING_BY_VARINT_TYPE` | 32-bit and 64-bit integers only |

#### Reference: `ConstantEncoding`

`ConstantEncoding.h` is the simplest encoding and a good starting template. It stores a single value and replays it for every row.

### Step 3: Define Nested Stream Identifiers (if applicable)

**File:** `velox/dwio/nimble/encodings/selection/EncodingIdentifier.h`

If your encoding has nested sub-streams, add a struct:

```cpp
struct EncodingIdentifiers {
  // ... existing ...
  struct MyEncoding {
    static constexpr NestedEncodingIdentifier ChildStream = 0;
    static constexpr NestedEncodingIdentifier AnotherStream = 1;
  };
};
```

Skip this step if your encoding has no nested encodings.

### Step 4: Register in `EncodingFactory` (Read + Write Path)

**File:** `velox/dwio/nimble/encodings/common/EncodingFactory.cpp`

Add the include at the top:

```cpp
#include "velox/dwio/nimble/encodings/MyEncoding.h"
```

Add a case to the **read path** (`create()` method):

```cpp
case EncodingType::MyEncoding: {
  RETURN_ENCODING_BY_LEAF_TYPE(MyEncoding, dataType);
}
```

Add a case to the **write path** (`encode<T>()` template):

```cpp
case EncodingType::MyEncoding: {
  return MyEncoding<T>::encode(selection, values, buffer, options);
}
```

### Step 5: Register in `EncodingUtils.h` (Selective Reader Dispatch)

**File:** `velox/dwio/nimble/encodings/common/EncodingUtils.h`

Add the include:

```cpp
#include "velox/dwio/nimble/encodings/MyEncoding.h"
```

Add a case to `encodingTypeDispatchNonString<T>()` (and/or `encodingTypeDispatchString()` if string is supported):

```cpp
case EncodingType::MyEncoding:
  return static_cast<MyEncoding<T>&>(encoding)
      .readWithVisitor(visitor, params);
```

### Step 6: Add Size Estimation (for Encoding Selection)

**File:** `velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h`

Add estimation logic to the appropriate private helper(s):

- `estimateNumericSize<T>()` — for numeric types
- `estimateBoolSize()` — for bool
- `estimateStringSize()` — for string

Each method has a switch on `EncodingType`. Add a case that returns the estimated encoded size in bytes, or `std::nullopt` if the encoding doesn't support the data type:

```cpp
case EncodingType::MyEncoding: {
  // Return estimated encoded size based on statistics
  return someEstimate;
}
```

### Step 7: Enable in Encoding Selection Policy

**File:** `velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.cpp`

Add the encoding to `ManualEncodingSelectionPolicyFactory`:

1. **`defaultEncodingReadFactors()`** — add the encoding with a read cost factor (1.0 = baseline, lower = preferred) if it should be selected by default:

```cpp
static std::vector<std::pair<EncodingType, float>> defaultEncodingReadFactors() {
  return {
      // ... existing ...
      {EncodingType::MyEncoding, 1.0},
  };
}
```

2. **`possibleEncodings()`** — add the encoding type if it should be accepted in `parseEncodingReadFactors()` runtime config:

```cpp
static std::vector<EncodingType> possibleEncodings() {
  return {
      // ... existing ...
      EncodingType::MyEncoding,
  };
}
```

**Note:** If the encoding is special-purpose (like `Nullable`, `Sentinel`, or `Delta`), you may choose NOT to add it to the general selection and instead trigger it explicitly via `EncodingLayoutTree` or a custom policy.

### Step 8: Update Build Files

**File:** `velox/dwio/nimble/encodings/BUCK`

Add to the `encodings` target:

```python
cpp_library(
    name = "encodings",
    srcs = [
        # ... existing ...
        "MyEncoding.cpp",  # if you have a .cpp file
    ],
    headers = [
        # ... existing ...
        "MyEncoding.h",
    ],
    # ...
)
```

**File:** `velox/dwio/nimble/encodings/CMakeLists.txt` (for OSS build)

Add `.cpp` file to the `nimble_encodings` library if applicable.

### Step 9: Write Tests

Create `velox/dwio/nimble/encodings/tests/MyEncodingTest.cpp`.

**File:** `velox/dwio/nimble/encodings/tests/TestUtils.h`

Add an `EncodingTypeTraits` specialization so the test utility knows your encoding's type:

```cpp
template <typename T>
struct EncodingTypeTraits<MyEncoding<T>> {
  static constexpr EncodingType encodingType = EncodingType::MyEncoding;
};
```

**Test pattern** (see `ConstantEncodingTest.cpp` as reference):

```cpp
#include <gtest/gtest.h>
#include "velox/dwio/nimble/encodings/MyEncoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;

TEST(MyEncodingTests, basicRoundtrip) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  nimble::Buffer buffer{*pool};

  std::vector<int32_t> data{/* test data */};

  auto encoding =
      nimble::test::Encoder<nimble::MyEncoding<int32_t>>::createEncoding(
          *pool, data, buffer);

  EXPECT_EQ(encoding->encodingType(), nimble::EncodingType::MyEncoding);
  EXPECT_EQ(encoding->rowCount(), data.size());

  std::vector<int32_t> result(data.size());
  encoding->materialize(data.size(), result.data());
  EXPECT_EQ(result, data);
}
```

**File:** `velox/dwio/nimble/encodings/tests/BUCK`

Add the test file to the `tests` target's `srcs` list.

### Step 10: Lint and Test

```bash
arc f                # format
arc lint -a          # lint + autodeps

buck build fbcode//velox/dwio/nimble/encodings:encodings
buck test fbcode//velox/dwio/nimble/encodings/tests:tests
```

## Enabling and Disabling Encodings

Once an encoding is registered in the factory (Steps 4-5), it can always **decode** data written with it. Enabling and disabling controls whether the encoding is **selected at write time**. There are several layers of control, from broad defaults to per-stream overrides.

### Encoding Selection Candidates (Default Behavior)

The `ManualEncodingSelectionPolicyFactory` in `encodings/selection/EncodingSelectionPolicy.cpp` controls which encodings are candidates during write. Two lists control the candidate surface:

- **`defaultEncodingReadFactors()`** — encoding + cost weight pairs. Only encodings in this list are considered. A lower read factor makes an encoding more favorable.
- **`possibleEncodings()`** — private list used for string-based config parsing.

**To disable an encoding globally:** Remove it from `defaultEncodingReadFactors()`. The encoding still works for decoding existing data — only new writes stop using it by default.

**To make an encoding configurable but not default:** Keep it in `possibleEncodings()` and leave it out of `defaultEncodingReadFactors()`. For example, `Delta`, `Prefix`, and experimental encodings can be triggered explicitly without being selected by default.

### Custom Read Factors at Write Time

The `ManualEncodingSelectionPolicyFactory` constructor accepts custom read factors, allowing callers to override the default candidate list per writer instance:

```cpp
// Only consider Trivial and RLE, with RLE strongly preferred
ManualEncodingSelectionPolicyFactory factory{
    {{EncodingType::Trivial, 1.0}, {EncodingType::RLE, 0.5}}};

writerOptions.encodingSelectionPolicyCreator =
    [factory](DataType dataType) {
      return factory.createPolicy(dataType);
    };
```

Encodings not in the provided list are excluded from selection. This is the primary way to disable encodings for a specific writer without changing the library defaults.

### String-Based Configuration (Parseable Read Factors)

`ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors()` parses a semicolon-delimited string of `EncodingType=factor` pairs:

```cpp
auto factors = ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
    "Trivial=0.7;RLE=1.0;Dictionary=1.0");
ManualEncodingSelectionPolicyFactory factory{factors};
```

This enables runtime configuration via flags or config files. Only encodings listed in `possibleEncodings()` are accepted — unknown names cause an error.

### Forcing a Specific Encoding via `EncodingLayoutTree`

`EncodingLayoutTree` (in `encodings/EncodingLayoutTree.h`) captures a tree of encoding decisions that can be replayed on subsequent writes, bypassing the selection policy entirely. This is used for:

- **Encoding layout training** — write a file, capture the selected encodings, replay them for consistent encoding across files.
- **Forcing specific encodings per stream** — each node in the tree maps stream identifiers to `EncodingLayout` objects that specify the exact `EncodingType` and compression.

```cpp
// Build a tree that forces RLE on the scalar stream
EncodingLayoutTree tree{
    Kind::Scalar,
    {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
      EncodingLayout{EncodingType::RLE, {}, CompressionType::Zstd}}},
    "my_column"};

writerOptions.encodingLayoutTree = std::move(tree);
```

When an `EncodingLayoutTree` is set, the writer uses the specified encodings directly instead of running the selection policy. Streams not covered by the tree fall back to normal selection.

### Custom Encoding Selection Policy

For full control, implement a custom `EncodingSelectionPolicy<T>` and wire it through `WriterOptions::encodingSelectionPolicyCreator`:

```cpp
writerOptions.encodingSelectionPolicyCreator =
    [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
  // Return a custom policy that implements select() and selectNullable()
  UNIQUE_PTR_FACTORY(dataType, MyCustomPolicy);
};
```

The custom policy's `select()` method receives data statistics and returns an `EncodingSelectionResult` with the chosen `EncodingType`. This allows arbitrary logic: ML models, workload-specific heuristics, or A/B testing between encodings.

### Summary of Control Layers

| Layer | Scope | How |
|-------|-------|-----|
| Library defaults | All writers | Edit `defaultEncodingReadFactors()` in `EncodingSelectionPolicy.cpp` |
| Per-writer read factors | Single writer | Pass custom read factors to `ManualEncodingSelectionPolicyFactory` constructor |
| Runtime config string | Single writer | Use `parseEncodingReadFactors("Trivial=0.7;RLE=1.0")` from flags/config |
| Encoding layout tree | Per-stream | Set `writerOptions.encodingLayoutTree` to force specific encodings |
| Custom policy | Single writer | Implement `EncodingSelectionPolicy<T>` and set `encodingSelectionPolicyCreator` |

**Important:** Disabling an encoding at the selection layer only affects **writes**. The read path (`EncodingFactory::create()`) always supports all registered encodings, ensuring backward compatibility with existing files.

## Integration Checklist

| # | File | What to do |
|---|------|------------|
| 1 | `common/Types.h` | Add `EncodingType` enum value |
| 2 | `common/Types.cpp` | Add `toString()` case |
| 3 | `encodings/MyEncoding.h` (+`.cpp`) | Implement the encoding |
| 4 | `encodings/selection/EncodingIdentifier.h` | Add nested stream IDs (if applicable) |
| 5 | `encodings/common/EncodingFactory.cpp` | Add read (`create()`) and write (`encode()`) dispatch |
| 6 | `encodings/common/EncodingUtils.h` | Add `readWithVisitor` dispatch |
| 7 | `encodings/selection/EncodingSizeEstimation.h` | Add size estimation |
| 8 | `encodings/selection/EncodingSelectionPolicy.h` | Add to candidate list and read factors |
| 9 | `encodings/BUCK` + `encodings/CMakeLists.txt` | Add to build |
| 10 | `encodings/tests/TestUtils.h` | Add `EncodingTypeTraits` specialization |
| 11 | `encodings/tests/MyEncodingTest.cpp` | Write tests |
| 12 | `encodings/tests/BUCK` | Add test to build |
