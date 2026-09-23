# Nimble Encoding Selection Flow: How EncodingSelectionPolicy Creates the Encoding Tree

This document explains how `EncodingSelectionPolicy` is used on the write path to create the final Encoding tree for data streams in Nimble.

## 🎯 Overview

Nimble uses a **top-down greedy approach** to build an encoding tree:
1. Inspect top-level data and pick an encoding
2. If the encoding is nested (contains sub-streams), break down the data
3. Recursively apply the same algorithm to each nested encoding
4. Once an encoding is selected at a level, commit to it (no backtracking)

## 📝 Key Components

### 1. **EncodingSelectionPolicy** (Policy Interface)
- **Base class**: `EncodingSelectionPolicy<T>`
- **Purpose**: Pluggable policy for selecting the best encoding for given data
- **Key methods**:
  - `select(values, statistics)` → Returns `EncodingSelectionResult`
  - `selectNullable(values, nulls, statistics)` → For nullable data
  - `createImpl(encodingType, identifier, dataType)` → Creates nested policy

### 2. **EncodingSelection** (Selection Context)
- **Purpose**: Passed to `Encoding::encode()` methods
- **Contains**:
  - Selected encoding type
  - Pre-calculated statistics
  - Reference to the selection policy
  - Compression policy factory
- **Key method**:
  - `encodeNested<NestedT>(identifier, values, buffer)` → Triggers nested encoding

### 3. **EncodingFactory** (Orchestrator)
- **Purpose**: Entry point for encoding data
- **Key methods**:
  - `encode<T>(policy, values, buffer)` → Public API
  - `encode<T>(selection, values, buffer)` → Internal dispatcher (private)

### 4. **Statistics** (Data Analysis)
- **Purpose**: Provides rich statistical information about the data
- **Includes**: Unique counts, min/max repeat, value distribution, etc.
- **Used by**: Policy to make informed encoding decisions

## 🔄 Complete Write Path Flow

### Phase 1: Top-Level Encoding Selection

```cpp
// User calls EncodingFactory::encode
auto encoded = EncodingFactory::encode<uint32_t>(
    std::make_unique<ManualEncodingSelectionPolicy<uint32_t>>(...),
    values,
    buffer);
```

**Step-by-step execution**:

1. **Create Statistics** (`EncodingFactory::encode`, line 205)
   ```cpp
   auto statistics = Statistics<physicalType>::create(physicalValues);
   ```

2. **Call Policy Selection** (line 206)
   ```cpp
   auto selectionResult = selectorPolicy->select(physicalValues, statistics);
   ```

3. **Policy Decision** (e.g., `ManualEncodingSelectionPolicy::select`, line 160-280)
   - Iterates through candidate encodings (Constant, Trivial, Dictionary, RLE, etc.)
   - Estimates size for each encoding using `EncodingSizeEstimation`
   - Applies read factors (performance weights)
   - Calculates cost: `size * readFactor`
   - Selects encoding with minimal cost
   - Returns `EncodingSelectionResult` with:
     - Selected `EncodingType` (e.g., `EncodingType::Dictionary`)
     - Compression policy factory

4. **Create EncodingSelection** (line 207-210)
   ```cpp
   EncodingSelection<physicalType> selection{
       std::move(selectionResult),
       std::move(statistics),
       std::move(selectorPolicy)
   };
   ```

5. **Dispatch to Encoding** (line 211-212, then 234-302)
   ```cpp
   return EncodingFactory::encode<T>(std::move(selection), physicalValues, buffer);
   ```

   This dispatches to the selected encoding's `encode()` method:
   ```cpp
   switch (selection.encodingType()) {
       case EncodingType::Dictionary:
           return DictionaryEncoding<T>::encode(selection, castedValues, buffer);
       case EncodingType::RLE:
           return RLEEncoding<T>::encode(selection, castedValues, buffer);
       // ... other encodings
   }
   ```

### Phase 2: Nested Encoding (Example: DictionaryEncoding)

**DictionaryEncoding needs two nested streams**:
- **Alphabet**: Unique values (e.g., `["apple", "banana", "cherry"]`)
- **Indices**: Positions mapping (e.g., `[0, 1, 0, 2, 1]`)

**Execution** (`DictionaryEncoding::encode`, line 197-285):

1. **Extract Alphabet and Indices** (line 206-261)
   ```cpp
   Vector<physicalType> alphabet;  // ["apple", "banana", "cherry"]
   Vector<uint32_t> indices;       // [0, 1, 0, 2, 1]
   ```

2. **Encode Nested Alphabet** (line 267-269)
   ```cpp
   std::string_view serializedAlphabet =
       selection.template encodeNested<physicalType>(
           EncodingIdentifiers::Dictionary::Alphabet,
           {alphabet},
           tempBuffer);
   ```

   **What happens inside `encodeNested`** (`EncodingSelection::encodeNested`, line 264-283):

   a. **Create Child Policy** (line 270-273)
      ```cpp
      auto nestedPolicy = selectionPolicy_->template create<physicalType>(
          EncodingType::Dictionary,
          EncodingIdentifiers::Dictionary::Alphabet);
      ```

      This calls `ManualEncodingSelectionPolicy::createImpl` (line 130-158):
      - Filters out parent encoding from candidate list (prevents infinite recursion)
      - Creates new policy instance with same configuration

   b. **Create Statistics for Nested Data** (line 274)
      ```cpp
      auto statistics = Statistics<physicalType>::create(alphabet);
      ```

   c. **Select Encoding for Alphabet** (line 275)
      ```cpp
      auto selectionResult = nestedPolicy->select(alphabet, statistics);
      ```
      - Policy might select `TrivialEncoding` for small alphabets
      - Or `FixedBitWidthEncoding` for numeric alphabets

   d. **Recursively Encode** (line 276-282)
      ```cpp
      return EncodingFactory::encode<physicalType>(
          EncodingSelection<physicalType>{
              std::move(selectionResult),
              std::move(statistics),
              std::move(nestedPolicy)
          },
          alphabet,
          buffer);
      ```
      - **Recursive call** back to Phase 1!
      - If selected encoding is also nested (e.g., RLE), the process repeats

3. **Encode Nested Indices** (line 270-272)
   ```cpp
   std::string_view serializedIndices =
       selection.template encodeNested<uint32_t>(
           EncodingIdentifiers::Dictionary::Indices,
           {indices},
           tempBuffer);
   ```
   - Same recursive process
   - Policy might select `FixedBitWidthEncoding` or `RLEEncoding` for indices

4. **Combine Results** (line 274-284)
   ```cpp
   // Layout: [Prefix | AlphabetSize | Alphabet bytes | Indices bytes]
   char* reserved = buffer.reserve(encodingSize);
   Encoding::serializePrefix(...);
   encoding::writeUint32(serializedAlphabet.size(), pos);
   encoding::writeBytes(serializedAlphabet, pos);
   encoding::writeBytes(serializedIndices, pos);
   ```

### Phase 3: Compression (Applied at Each Level)

After encoding is complete, compression is applied:

1. **Get Compression Policy** (`EncodingSelection::compressionPolicy()`)
   ```cpp
   auto policy = selectionResult_.compressionPolicyFactory();
   ```

2. **Compression Information** (from `ConfiguredCompressionPolicy::config()`)
   ```cpp
   CompressionConfig info{
       .compressionType = CompressionType::MetaInternal,
       .compressionLevel = 4,
       // ... other parameters
   };
   ```

3. **Apply Compression** (in Encoding implementations)
   - Compresses the serialized bytes
   - Calls `compressionPolicy->shouldAccept(type, uncompressed, compressed)`
   - If compression ratio is good (< 98% of original), uses compressed version
   - Otherwise, keeps uncompressed version

## 📊 Example: Full Encoding Tree

**Input**: Array of strings with duplicates
```
["apple", "banana", "apple", "cherry", "banana"]
```

**Resulting Encoding Tree**:
```
DictionaryEncoding<string>
├── Alphabet: TrivialEncoding<string>
│   └── Data: ["apple", "banana", "cherry"]
│       └── Compression: MetaInternal (if beneficial)
└── Indices: FixedBitWidthEncoding<uint32_t>
    └── Data: [0, 1, 0, 2, 1]
        └── Compression: MetaInternal (if beneficial)
```

**How it was built**:
1. **Top level**: Policy selected `DictionaryEncoding` (best cost)
2. **Alphabet nested**: Policy selected `TrivialEncoding` (small unique set)
3. **Indices nested**: Policy selected `FixedBitWidthEncoding` (fits in 2 bits: 0-2)
4. **Compression**: Applied at each level where beneficial

## 🎛️ Policy Implementations

### ManualEncodingSelectionPolicy (Most Common)
- Uses **cost-based selection**: `cost = estimatedSize * readFactor`
- **Read factors** weight encodings by decode performance
  - Trivial: 0.7 (fast to decode + compression benefits)
  - FixedBitWidth: 0.9 (faster than variable-width)
  - Dictionary, RLE, MainlyConstant: 1.0 (standard)
- Excludes parent encoding when creating nested policy (prevents recursion)

### LearnedEncodingSelectionPolicy
- Uses **ML model** to predict encoding choice
- Currently simplified (mostly picks Trivial)
- Future: Multi-class prediction for all encoding types

### ReplayedEncodingSelectionPolicy
- Uses **pre-captured EncodingLayout** from previous encoding
- Useful for testing or reproducing exact encoding trees
- Creates child policies from layout's children

## 🔑 Key Design Patterns

### 1. **Type Erasure** (`EncodingSelectionPolicyBase`)
- Problem: Nested encodings have different types than parent (e.g., `string` parent, `uint32_t` indices)
- Solution: Non-templated base class allows runtime type switching
- `createImpl(DataType)` returns base pointer, caller casts to correct `EncodingSelectionPolicy<NestedT>`

### 2. **Factory Method Pattern**
- `EncodingFactory::encode()` dispatches to correct encoding based on type
- Each encoding implements `static encode()` method
- Consistent interface across all encoding types

### 3. **Recursive Composition**
- Encodings can contain other encodings
- Selection process naturally recurses through the tree
- Each level has its own policy instance (possibly filtered)

### 4. **Strategy Pattern**
- CompressionPolicy is pluggable
- EncodingSelectionPolicy is pluggable
- Easy to add new policies without changing encoding implementations

## 📐 Class Hierarchy

```
EncodingSelectionPolicyBase (non-templated)
├── createImpl(encodingType, identifier, dataType) = 0
└── [enables type-erased policy creation]

EncodingSelectionPolicy<T> : EncodingSelectionPolicyBase
├── select(values, statistics) = 0
├── selectNullable(values, nulls, statistics) = 0
└── [main policy interface]

ManualEncodingSelectionPolicy<T> : EncodingSelectionPolicy<T>
├── Implements cost-based selection
├── Uses EncodingSizeEstimation for each encoding
└── Filters parent encoding in createImpl()

LearnedEncodingSelectionPolicy<T> : EncodingSelectionPolicy<T>
├── Uses ML model for prediction
└── Future: Multi-class encoding prediction

ReplayedEncodingSelectionPolicy<T> : EncodingSelectionPolicy<T>
├── Replays from EncodingLayout
└── Used for testing/debugging
```

## 🔍 Code References

### Entry Points
- **User API**: `EncodingFactory::encode<T>(policy, values, buffer)` (EncodingFactory.cpp:199)
- **Internal**: `EncodingFactory::encode<T>(selection, values, buffer)` (EncodingFactory.cpp:234)

### Policy Selection
- **Manual Policy**: `ManualEncodingSelectionPolicy::select()` (EncodingSelectionPolicy.h:160)
- **Learned Policy**: `LearnedEncodingSelectionPolicy::select()` (EncodingSelectionPolicy.h:454)
- **Replayed Policy**: `ReplayedEncodingSelectionPolicy::select()` (EncodingSelectionPolicy.h:609)

### Nested Encoding
- **Trigger**: `EncodingSelection::encodeNested<NestedT>()` (EncodingSelection.h:264)
- **Example usage**: `DictionaryEncoding::encode()` (DictionaryEncoding.h:267-272)

### Policy Creation
- **Base method**: `EncodingSelectionPolicyBase::create<NestedT>()` (EncodingSelection.h:211)
- **Implementation**: Policy-specific `createImpl()` methods

## 💡 Summary

The **EncodingSelectionPolicy** system enables Nimble to:
1. **Automatically select optimal encodings** based on data characteristics
2. **Build nested encoding trees** recursively and efficiently
3. **Apply compression intelligently** at each level
4. **Support pluggable policies** (manual, learned, replayed)
5. **Make type-safe selections** across different data types

The key insight is the **recursive nature** of the process:
- Each encoding selection can trigger nested selections
- Each nested selection has its own policy instance
- The tree is built top-down with no backtracking
- Statistics guide intelligent decisions at each level

This architecture allows Nimble to achieve excellent compression ratios while maintaining fast decode performance.
