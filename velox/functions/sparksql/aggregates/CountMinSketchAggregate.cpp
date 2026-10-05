/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/functions/sparksql/aggregates/CountMinSketchAggregate.h"

#include <bit>
#include <cmath>
#include <optional>
#include <string_view>

#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/Murmur3Hash.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate::sparksql {

namespace {

// Java-compatible linear congruential PRNG matching java.util.Random.
// Used to generate hashA seeds identically to Spark's CountMinSketchImpl.
class JavaRandom {
 public:
  explicit JavaRandom(int64_t seed) {
    seed_ = (seed ^ 0x5DEECE66DLLU) & ((1LLU << 48) - 1);
  }

  // Returns a value in [0, bound), matching java.util.Random.nextInt(bound).
  int32_t nextInt(int32_t bound) {
    VELOX_DCHECK_GT(bound, 0);
    // If bound is a power of two, use the fast path.
    if ((bound & (bound - 1)) == 0) {
      return static_cast<int32_t>(
          (static_cast<int64_t>(bound) * static_cast<int64_t>(next(31))) >> 31);
    }
    int32_t bits, val;
    do {
      bits = next(31);
      val = bits % bound;
      // Reproduce Java's int32 wraparound rejection with unsigned arithmetic
      // to avoid signed-overflow UB when bound is near INT32_MAX.
    } while (static_cast<int32_t>(
                 static_cast<uint32_t>(bits) - static_cast<uint32_t>(val) +
                 static_cast<uint32_t>(bound - 1)) < 0);
    return val;
  }

 private:
  int32_t next(int32_t bits) {
    seed_ = (seed_ * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    return static_cast<int32_t>(seed_ >> (48 - bits));
  }

  uint64_t seed_;
};

// Big-endian serialization helpers matching Java's DataOutputStream.
inline void writeBigEndianInt(char*& buf, int32_t value) {
  uint32_t v = static_cast<uint32_t>(value);
  buf[0] = static_cast<char>((v >> 24) & 0xFF);
  buf[1] = static_cast<char>((v >> 16) & 0xFF);
  buf[2] = static_cast<char>((v >> 8) & 0xFF);
  buf[3] = static_cast<char>(v & 0xFF);
  buf += 4;
}

inline void writeBigEndianLong(char*& buf, int64_t value) {
  uint64_t v = static_cast<uint64_t>(value);
  buf[0] = static_cast<char>((v >> 56) & 0xFF);
  buf[1] = static_cast<char>((v >> 48) & 0xFF);
  buf[2] = static_cast<char>((v >> 40) & 0xFF);
  buf[3] = static_cast<char>((v >> 32) & 0xFF);
  buf[4] = static_cast<char>((v >> 24) & 0xFF);
  buf[5] = static_cast<char>((v >> 16) & 0xFF);
  buf[6] = static_cast<char>((v >> 8) & 0xFF);
  buf[7] = static_cast<char>(v & 0xFF);
  buf += 8;
}

inline int32_t readBigEndianInt(const char*& buf) {
  uint32_t v = (static_cast<uint32_t>(static_cast<uint8_t>(buf[0])) << 24) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[1])) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buf[2])) << 8) |
      static_cast<uint32_t>(static_cast<uint8_t>(buf[3]));
  buf += 4;
  return static_cast<int32_t>(v);
}

inline int64_t readBigEndianLong(const char*& buf) {
  uint64_t v = (static_cast<uint64_t>(static_cast<uint8_t>(buf[0])) << 56) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[1])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[2])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[3])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[4])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[5])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buf[6])) << 8) |
      static_cast<uint64_t>(static_cast<uint8_t>(buf[7]));
  buf += 8;
  return std::bit_cast<int64_t>(v);
}

inline int64_t wrappingAdd(int64_t left, int64_t right) {
  return std::bit_cast<int64_t>(
      static_cast<uint64_t>(left) + static_cast<uint64_t>(right));
}

static constexpr int64_t kPrimeModulus = (1LL << 31) - 1;

int64_t serializedSketchSize(int32_t depth, int32_t width) {
  // Version(4) + TotalCount(8) + Depth(4) + Width(4) +
  // HashA(depth*8) + Table(depth*width*8)
  return 20 + static_cast<int64_t>(depth) * 8 +
      static_cast<int64_t>(depth) * width * 8;
}

// Serializes an all-zero sketch without modifying accumulator state.
void serializeEmptySketch(
    char* output,
    int32_t depth,
    int32_t width,
    const std::vector<int64_t>& hashA) {
  char* buffer = output;
  writeBigEndianInt(buffer, 1);
  writeBigEndianLong(buffer, 0);
  writeBigEndianInt(buffer, depth);
  writeBigEndianInt(buffer, width);
  for (const auto hash : hashA) {
    writeBigEndianLong(buffer, hash);
  }
  for (int64_t i = 0; i < static_cast<int64_t>(depth) * width; ++i) {
    writeBigEndianLong(buffer, 0);
  }
}

// Accumulator storing the CountMinSketch state.
// The binary format matches Spark's CountMinSketchImpl (V1, big-endian).
struct CountMinSketchAccumulator {
  explicit CountMinSketchAccumulator(HashStringAllocator* allocator)
      : hashA_{StlAllocator<int64_t>(allocator)},
        table_{StlAllocator<int64_t>(allocator)} {}

  bool initialized() const {
    return depth_ > 0;
  }

  void init(int32_t depth, int32_t width, const std::vector<int64_t>& hashA) {
    if (initialized()) {
      return;
    }
    depth_ = depth;
    width_ = width;
    totalCount_ = 0;
    hashA_.resize(depth);
    for (int32_t i = 0; i < depth; ++i) {
      hashA_[i] = hashA[i];
    }
    table_.resize(static_cast<size_t>(depth) * width, 0);
  }

  // Hash function for integral types matching Spark's CountMinSketchImpl.hash.
  // Uses unsigned arithmetic to avoid signed overflow UB (Java wraps; C++ UB).
  int32_t hashLong(int64_t item, int32_t i) const {
    // Perform every step in uint64_t to reproduce Java's wrapping long
    // arithmetic; signed int64_t overflow (in the multiply and the addition
    // below) is undefined behavior in C++.
    uint64_t hash =
        static_cast<uint64_t>(hashA_[i]) * static_cast<uint64_t>(item);
    hash += hash >> 32;
    hash &= static_cast<uint64_t>(kPrimeModulus);
    return static_cast<int32_t>(hash % static_cast<uint64_t>(width_));
  }

  // Hash function for binary/string types matching Spark's
  // CountMinSketchImpl.getHashBuckets.
  // Uses unsigned arithmetic to avoid signed overflow UB.
  void getHashBuckets(const char* data, int32_t length, int32_t* buckets)
      const {
    int32_t hash1 = static_cast<int32_t>(
        ::facebook::velox::functions::sparksql::SparkMurmur3Hash::hashBytes(
            data, length, 0));
    int32_t hash2 = static_cast<int32_t>(
        ::facebook::velox::functions::sparksql::SparkMurmur3Hash::hashBytes(
            data, length, static_cast<uint32_t>(hash1)));
    for (int32_t i = 0; i < depth_; ++i) {
      auto combined = static_cast<int32_t>(
          static_cast<uint32_t>(hash1) +
          static_cast<uint32_t>(i) * static_cast<uint32_t>(hash2));
      buckets[i] = std::abs(combined % width_);
    }
  }

  void addLong(int64_t item) {
    for (int32_t i = 0; i < depth_; ++i) {
      int32_t col = hashLong(item, i);
      auto& count = table_[static_cast<size_t>(i) * width_ + col];
      count = wrappingAdd(count, 1);
    }
    totalCount_ = wrappingAdd(totalCount_, 1);
  }

  void addBinary(const char* data, int32_t length) {
    if (depth_ <= 64) {
      int32_t buckets[64];
      getHashBuckets(data, length, buckets);
      for (int32_t i = 0; i < depth_; ++i) {
        auto& count = table_[static_cast<size_t>(i) * width_ + buckets[i]];
        count = wrappingAdd(count, 1);
      }
    } else {
      std::vector<int32_t> buckets(depth_);
      getHashBuckets(data, length, buckets.data());
      for (int32_t i = 0; i < depth_; ++i) {
        auto& count = table_[static_cast<size_t>(i) * width_ + buckets[i]];
        count = wrappingAdd(count, 1);
      }
    }
    totalCount_ = wrappingAdd(totalCount_, 1);
  }

  int64_t serializedSize() const {
    return serializedSketchSize(depth_, width_);
  }

  void serialize(char* output) const {
    char* buf = output;
    writeBigEndianInt(buf, 1); // Version V1
    writeBigEndianLong(buf, totalCount_);
    writeBigEndianInt(buf, depth_);
    writeBigEndianInt(buf, width_);
    for (int32_t i = 0; i < depth_; ++i) {
      writeBigEndianLong(buf, hashA_[i]);
    }
    for (int32_t i = 0; i < depth_; ++i) {
      for (int32_t j = 0; j < width_; ++j) {
        writeBigEndianLong(buf, table_[static_cast<size_t>(i) * width_ + j]);
      }
    }
  }

  void mergeWith(const StringView& serialized) {
    const char* buf = serialized.data();

    // Validate minimum header size.
    VELOX_USER_CHECK_GE(
        serialized.size(),
        20,
        "CountMinSketch serialized data too small: {} bytes",
        serialized.size());

    int32_t version = readBigEndianInt(buf);
    VELOX_USER_CHECK_EQ(
        version, 1, "Unexpected CountMinSketch version: {}", version);

    int64_t otherTotalCount = readBigEndianLong(buf);
    int32_t otherDepth = readBigEndianInt(buf);
    int32_t otherWidth = readBigEndianInt(buf);

    VELOX_USER_CHECK_GT(otherDepth, 0, "CountMinSketch depth must be positive");
    VELOX_USER_CHECK_GT(otherWidth, 0, "CountMinSketch width must be positive");

    // Validate full serialized size with overflow-safe computation.
    int64_t hashASize = static_cast<int64_t>(otherDepth) * 8;
    int64_t tableSize = static_cast<int64_t>(otherDepth) * otherWidth;
    // Bound the dimensions so tableSize * 8 (and the subsequent table
    // allocation) cannot overflow int64. The truncation check below then
    // transitively bounds the allocation by the serialized input size.
    VELOX_USER_CHECK_LE(
        tableSize,
        (std::numeric_limits<int64_t>::max() - 20 - hashASize) / 8,
        "CountMinSketch dimensions too large: {} x {}",
        otherDepth,
        otherWidth);
    int64_t expectedSize = 20 + hashASize + tableSize * 8;
    VELOX_USER_CHECK_GE(
        static_cast<int64_t>(serialized.size()),
        expectedSize,
        "CountMinSketch serialized data truncated: {} < {} bytes",
        serialized.size(),
        expectedSize);

    if (!initialized()) {
      // First merge initializes this accumulator from the serialized data.
      depth_ = otherDepth;
      width_ = otherWidth;
      totalCount_ = otherTotalCount;
      hashA_.resize(otherDepth);
      for (int32_t i = 0; i < otherDepth; ++i) {
        hashA_[i] = readBigEndianLong(buf);
      }
      table_.resize(static_cast<size_t>(otherDepth) * otherWidth);
      for (int32_t i = 0; i < otherDepth; ++i) {
        for (int32_t j = 0; j < otherWidth; ++j) {
          table_[static_cast<size_t>(i) * otherWidth + j] =
              readBigEndianLong(buf);
        }
      }
      return;
    }

    VELOX_USER_CHECK_EQ(
        depth_, otherDepth, "Cannot merge CountMinSketch of different depth");
    VELOX_USER_CHECK_EQ(
        width_, otherWidth, "Cannot merge CountMinSketch of different width");

    // Read and validate hashA matches (same depth/width can have different
    // seeds — merging such sketches silently corrupts results).
    for (int32_t i = 0; i < otherDepth; ++i) {
      auto otherHash = readBigEndianLong(buf);
      VELOX_USER_CHECK_EQ(
          hashA_[i],
          otherHash,
          "Cannot merge CountMinSketch with different hash seeds at index {}",
          i);
    }

    // Merge table counts.
    for (int32_t i = 0; i < depth_; ++i) {
      for (int32_t j = 0; j < width_; ++j) {
        auto& count = table_[static_cast<size_t>(i) * width_ + j];
        count = wrappingAdd(count, readBigEndianLong(buf));
      }
    }
    totalCount_ = wrappingAdd(totalCount_, otherTotalCount);
  }

  int32_t depth_{0};
  int32_t width_{0};
  int64_t totalCount_{0};
  std::vector<int64_t, StlAllocator<int64_t>> hashA_;
  std::vector<int64_t, StlAllocator<int64_t>> table_;
};

template <typename T>
T readConstantArgument(const VectorPtr& argument, std::string_view name) {
  VELOX_USER_CHECK_GT(
      argument->size(), 0, "{} argument requires at least one input row", name);
  SelectivityVector rows(argument->size());
  DecodedVector decoded(*argument, rows);
  const auto firstRow = rows.begin();
  VELOX_USER_CHECK(
      !decoded.isNullAt(firstRow), "{} argument must not be null", name);
  const auto value = decoded.valueAt<T>(firstRow);
  rows.applyToSelected([&](vector_size_t row) {
    VELOX_USER_CHECK(
        !decoded.isNullAt(row), "{} argument must not be null", name);
    VELOX_USER_CHECK(
        decoded.valueAt<T>(row) == value,
        "{} argument must be constant for all input rows",
        name);
  });
  return value;
}

class CountMinSketchAggregate : public exec::Aggregate {
 public:
  explicit CountMinSketchAggregate(const TypePtr& resultType)
      : Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(CountMinSketchAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  // Captures the constant eps, confidence and seed arguments before any data is
  // processed. This makes the sketch dimensions available even when the input
  // is empty, so a valid empty sketch can be emitted (Spark's count_min_sketch
  // is non-nullable and always serializes its buffer). For final aggregation
  // the raw inputs are the intermediate state only, so the constants are
  // absent; the dimensions are then recovered by merging intermediate states.
  void setConstantInputs(
      const std::vector<VectorPtr>& constantInputs) override {
    if (constantInputs.size() < 4 || constantInputs[1] == nullptr ||
        constantInputs[2] == nullptr || constantInputs[3] == nullptr) {
      return;
    }
    computeDimensions(constantInputs);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    rows.applyToSelected([&](vector_size_t row) {
      auto group = groups[row];
      auto* accumulator = value<CountMinSketchAccumulator>(group);
      // Initialize accumulator for every group that has rows, regardless
      // of whether this row's value is null. Matches Spark's behavior of
      // always returning a valid (empty) sketch for groups with only NULLs.
      if (!accumulator->initialized()) {
        auto tracker = trackRowSize(group);
        accumulator->init(depth_, width_, hashA_);
        clearNull(group);
      }
      if (decodedValue_.isNullAt(row)) {
        return;
      }
      addValue(accumulator, row);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_EQ(args.size(), 1);
    decodedIntermediate_.decode(*args[0], rows);
    rows.applyToSelected([&](auto row) {
      if (UNLIKELY(decodedIntermediate_.isNullAt(row))) {
        return;
      }
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      auto* accumulator = value<CountMinSketchAccumulator>(group);
      accumulator->mergeWith(serialized);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodeArguments(rows, args);
    auto tracker = trackRowSize(group);
    auto* accumulator = value<CountMinSketchAccumulator>(group);
    accumulator->init(depth_, width_, hashA_);
    clearNull(group);
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedValue_.isNullAt(row)) {
        return;
      }
      addValue(accumulator, row);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK_EQ(args.size(), 1);
    decodedIntermediate_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    auto* accumulator = value<CountMinSketchAccumulator>(group);
    rows.applyToSelected([&](auto row) {
      if (UNLIKELY(decodedIntermediate_.isNullAt(row))) {
        return;
      }
      auto serialized = decodedIntermediate_.valueAt<StringView>(row);
      accumulator->mergeWith(serialized);
    });
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractSketches(groups, numGroups, result);
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractSketches(groups, numGroups, result);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) CountMinSketchAccumulator(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto* group : groups) {
      auto* accumulator = value<CountMinSketchAccumulator>(group);
      accumulator->~CountMinSketchAccumulator();
    }
  }

 private:
  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    VELOX_USER_CHECK_GE(args.size(), 4);
    decodedValue_.decode(*args[0], rows);
    inputKind_ = args[0]->type()->kind();

    if (args[1]->size() > 0) {
      computeDimensions(args);
    }
  }

  // Extracts and validates the query-wide constant eps, confidence and seed
  // arguments (positions 1, 2 and 3). Derives the sketch dimensions on the
  // first input and verifies that subsequent input batches use the same
  // parameters.
  void computeDimensions(const std::vector<VectorPtr>& args) {
    const auto eps = readConstantArgument<double>(args[1], "eps");
    VELOX_USER_CHECK_GT(eps, 0.0, "eps must be positive");

    const auto confidence = readConstantArgument<double>(args[2], "confidence");
    VELOX_USER_CHECK_GT(confidence, 0.0, "confidence must be positive");
    VELOX_USER_CHECK_LT(confidence, 1.0, "confidence must be less than 1.0");

    int32_t seed{0};
    if (args[3]->type()->kind() == TypeKind::INTEGER) {
      seed = readConstantArgument<int32_t>(args[3], "seed");
    } else {
      seed =
          static_cast<int32_t>(readConstantArgument<int64_t>(args[3], "seed"));
    }

    if (epsilon_.has_value()) {
      VELOX_USER_CHECK_EQ(
          eps,
          epsilon_.value(),
          "eps argument must be constant for all input rows");
      VELOX_USER_CHECK_EQ(
          confidence,
          confidence_.value(),
          "confidence argument must be constant for all input rows");
      VELOX_USER_CHECK_EQ(
          seed,
          seed_.value(),
          "seed argument must be constant for all input rows");
      return;
    }
    epsilon_ = eps;
    confidence_ = confidence;
    seed_ = seed;

    // Compute depth and width matching Spark's CountMinSketchImpl.
    // Validate the width before narrowing to int32 to avoid undefined behavior
    // when 2 / eps is out of int32 range (very small eps), and division by
    // zero later when eps is +infinity (width 0).
    double widthDouble = std::ceil(2.0 / eps);
    VELOX_USER_CHECK(
        std::isfinite(widthDouble) && widthDouble >= 1.0 &&
            widthDouble <=
                static_cast<double>(std::numeric_limits<int32_t>::max()),
        "count_min_sketch width out of range for eps: {}",
        eps);
    width_ = static_cast<int32_t>(widthDouble);
    // Use log1p(-c) to match Spark's exact formula and rounding.
    depth_ = static_cast<int32_t>(
        std::ceil(-std::log1p(-confidence) / std::log(2.0)));
    const auto outputSize = serializedSketchSize(depth_, width_);
    VELOX_USER_CHECK_LE(
        outputSize,
        std::numeric_limits<int32_t>::max(),
        "count_min_sketch serialized size out of range: {} bytes",
        outputSize);

    // Initialize hashA using Java-compatible Random.
    JavaRandom rng(seed);
    hashA_.resize(depth_);
    for (int32_t i = 0; i < depth_; ++i) {
      hashA_[i] = rng.nextInt(std::numeric_limits<int32_t>::max());
    }
  }

  void addValue(CountMinSketchAccumulator* accumulator, vector_size_t row) {
    switch (inputKind_) {
      case TypeKind::TINYINT:
        accumulator->addLong(
            static_cast<int64_t>(decodedValue_.valueAt<int8_t>(row)));
        break;
      case TypeKind::SMALLINT:
        accumulator->addLong(
            static_cast<int64_t>(decodedValue_.valueAt<int16_t>(row)));
        break;
      case TypeKind::INTEGER:
        accumulator->addLong(
            static_cast<int64_t>(decodedValue_.valueAt<int32_t>(row)));
        break;
      case TypeKind::BIGINT:
        accumulator->addLong(decodedValue_.valueAt<int64_t>(row));
        break;
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: {
        auto sv = decodedValue_.valueAt<StringView>(row);
        accumulator->addBinary(sv.data(), static_cast<int32_t>(sv.size()));
        break;
      }
      default:
        VELOX_UNREACHABLE(
            "Unsupported type for count_min_sketch: {}",
            TypeKindName::toName(inputKind_));
    }
  }

  // Serializes initialized accumulators or aggregate-level empty sketches
  // without modifying group state.
  void extractSketches(char** groups, int32_t numGroups, VectorPtr* result)
      const {
    VELOX_CHECK(result);
    auto flatResult = (*result)->asUnchecked<FlatVector<StringView>>();
    flatResult->resize(numGroups);

    int64_t totalSize = 0;
    for (vector_size_t i = 0; i < numGroups; ++i) {
      const auto* accumulator = value<CountMinSketchAccumulator>(groups[i]);
      if (accumulator->initialized()) {
        totalSize += accumulator->serializedSize();
      } else if (depth_ != 0) {
        totalSize += serializedSketchSize(depth_, width_);
      }
    }

    char* rawBuffer = flatResult->getRawStringBufferWithSpace(totalSize);
    for (vector_size_t i = 0; i < numGroups; ++i) {
      const auto* accumulator = value<CountMinSketchAccumulator>(groups[i]);
      int64_t size;
      if (accumulator->initialized()) {
        size = accumulator->serializedSize();
        accumulator->serialize(rawBuffer);
      } else if (depth_ != 0) {
        size = serializedSketchSize(depth_, width_);
        serializeEmptySketch(rawBuffer, depth_, width_, hashA_);
      } else {
        // Final aggregation over no intermediate states has no dimensions.
        flatResult->setNull(i, true);
        continue;
      }
      flatResult->setNoCopy(
          i, StringView(rawBuffer, static_cast<int32_t>(size)));
      rawBuffer += size;
    }
  }

  DecodedVector decodedValue_;
  DecodedVector decodedIntermediate_;
  TypeKind inputKind_{TypeKind::INVALID};
  std::optional<double> epsilon_;
  std::optional<double> confidence_;
  std::optional<int32_t> seed_;
  int32_t depth_{0};
  int32_t width_{0};
  std::vector<int64_t> hashA_;
};

} // namespace

exec::AggregateRegistrationResult registerCountMinSketchAggregate(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  // Signatures for integral input types.
  for (const auto& inputType : {"tinyint", "smallint", "integer", "bigint"}) {
    for (const auto& seedType : {"integer", "bigint"}) {
      signatures.push_back(
          exec::AggregateFunctionSignatureBuilder()
              .argumentType(inputType)
              .constantArgumentType("double")
              .constantArgumentType("double")
              .constantArgumentType(seedType)
              .intermediateType("varbinary")
              .returnType("varbinary")
              .build());
    }
  }
  // Signatures for varchar input type.
  for (const auto& seedType : {"integer", "bigint"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .argumentType("varchar")
            .constantArgumentType("double")
            .constantArgumentType("double")
            .constantArgumentType(seedType)
            .intermediateType("varbinary")
            .returnType("varbinary")
            .build());
  }
  // Signatures for varbinary input type.
  for (const auto& seedType : {"integer", "bigint"}) {
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .argumentType("varbinary")
            .constantArgumentType("double")
            .constantArgumentType("double")
            .constantArgumentType(seedType)
            .intermediateType("varbinary")
            .returnType("varbinary")
            .build());
  }

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step /* step */,
          const std::vector<TypePtr>& /* argTypes */,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<CountMinSketchAggregate>(resultType);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
