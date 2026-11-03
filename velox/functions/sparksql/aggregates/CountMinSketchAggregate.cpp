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
#include <string_view>
#include <type_traits>

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/Murmur3Hash.h"
#include "velox/vector/ConstantVector.h"

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
    int32_t bits;
    int32_t value;
    do {
      bits = next(31);
      value = bits % bound;
      // Reproduce Java's int32 wraparound rejection with unsigned arithmetic
      // to avoid signed-overflow UB when bound is near INT32_MAX.
    } while (static_cast<int32_t>(
                 static_cast<uint32_t>(bits) - static_cast<uint32_t>(value) +
                 static_cast<uint32_t>(bound - 1)) < 0);
    return value;
  }

 private:
  int32_t next(int32_t numBits) {
    seed_ = (seed_ * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    return static_cast<int32_t>(seed_ >> (48 - numBits));
  }

  uint64_t seed_;
};

// Big-endian serialization helpers matching Java's DataOutputStream.
inline void writeBigEndianInt(char*& buffer, int32_t value) {
  uint32_t unsignedValue = static_cast<uint32_t>(value);
  buffer[0] = static_cast<char>((unsignedValue >> 24) & 0xFF);
  buffer[1] = static_cast<char>((unsignedValue >> 16) & 0xFF);
  buffer[2] = static_cast<char>((unsignedValue >> 8) & 0xFF);
  buffer[3] = static_cast<char>(unsignedValue & 0xFF);
  buffer += 4;
}

inline void writeBigEndianLong(char*& buffer, int64_t value) {
  uint64_t unsignedValue = static_cast<uint64_t>(value);
  buffer[0] = static_cast<char>((unsignedValue >> 56) & 0xFF);
  buffer[1] = static_cast<char>((unsignedValue >> 48) & 0xFF);
  buffer[2] = static_cast<char>((unsignedValue >> 40) & 0xFF);
  buffer[3] = static_cast<char>((unsignedValue >> 32) & 0xFF);
  buffer[4] = static_cast<char>((unsignedValue >> 24) & 0xFF);
  buffer[5] = static_cast<char>((unsignedValue >> 16) & 0xFF);
  buffer[6] = static_cast<char>((unsignedValue >> 8) & 0xFF);
  buffer[7] = static_cast<char>(unsignedValue & 0xFF);
  buffer += 8;
}

inline int32_t readBigEndianInt(const char*& buffer) {
  uint32_t unsignedValue =
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[0])) << 24) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[1])) << 16) |
      (static_cast<uint32_t>(static_cast<uint8_t>(buffer[2])) << 8) |
      static_cast<uint32_t>(static_cast<uint8_t>(buffer[3]));
  buffer += 4;
  return static_cast<int32_t>(unsignedValue);
}

inline int64_t readBigEndianLong(const char*& buffer) {
  uint64_t unsignedValue =
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[0])) << 56) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[1])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[2])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[3])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[4])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[5])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(buffer[6])) << 8) |
      static_cast<uint64_t>(static_cast<uint8_t>(buffer[7]));
  buffer += 8;
  return std::bit_cast<int64_t>(unsignedValue);
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

  void
  initialize(int32_t depth, int32_t width, const std::vector<int64_t>& hashA) {
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
  int32_t hashLong(int64_t item, int32_t hashIndex) const {
    // Perform every step in uint64_t to reproduce Java's wrapping long
    // arithmetic; signed int64_t overflow (in the multiply and the addition
    // below) is undefined behavior in C++.
    uint64_t hash =
        static_cast<uint64_t>(hashA_[hashIndex]) * static_cast<uint64_t>(item);
    hash += hash >> 32;
    hash &= static_cast<uint64_t>(kPrimeModulus);
    return static_cast<int32_t>(hash % static_cast<uint64_t>(width_));
  }

  // Hash function for binary/string types matching Spark's
  // CountMinSketchImpl.getHashBuckets.
  // Uses unsigned arithmetic to avoid signed overflow UB.
  void getHashBuckets(const char* data, int32_t length, int32_t* buckets)
      const {
    int32_t firstHash = static_cast<int32_t>(
        ::facebook::velox::functions::sparksql::SparkMurmur3Hash::hashBytes(
            data, length, 0));
    int32_t secondHash = static_cast<int32_t>(
        ::facebook::velox::functions::sparksql::SparkMurmur3Hash::hashBytes(
            data, length, static_cast<uint32_t>(firstHash)));
    for (int32_t i = 0; i < depth_; ++i) {
      auto combined = static_cast<int32_t>(
          static_cast<uint32_t>(firstHash) +
          static_cast<uint32_t>(i) * static_cast<uint32_t>(secondHash));
      buckets[i] = std::abs(combined % width_);
    }
  }

  void addLong(int64_t item) {
    for (int32_t i = 0; i < depth_; ++i) {
      int32_t column = hashLong(item, i);
      auto& count = table_[static_cast<size_t>(i) * width_ + column];
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
    char* buffer = output;
    writeBigEndianInt(buffer, 1); // Version V1
    writeBigEndianLong(buffer, totalCount_);
    writeBigEndianInt(buffer, depth_);
    writeBigEndianInt(buffer, width_);
    for (int32_t i = 0; i < depth_; ++i) {
      writeBigEndianLong(buffer, hashA_[i]);
    }
    for (int32_t i = 0; i < depth_; ++i) {
      for (int32_t j = 0; j < width_; ++j) {
        writeBigEndianLong(buffer, table_[static_cast<size_t>(i) * width_ + j]);
      }
    }
  }

  void mergeWith(const StringView& serialized) {
    const char* buffer = serialized.data();

    // Validate minimum header size.
    VELOX_USER_CHECK_GE(
        serialized.size(), 20, "CountMinSketch serialized data too small");

    int32_t version = readBigEndianInt(buffer);
    VELOX_USER_CHECK_EQ(version, 1, "Unexpected CountMinSketch version");

    int64_t otherTotalCount = readBigEndianLong(buffer);
    int32_t otherDepth = readBigEndianInt(buffer);
    int32_t otherWidth = readBigEndianInt(buffer);

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
        "CountMinSketch serialized data truncated");

    if (!initialized()) {
      // First merge initializes this accumulator from the serialized data.
      depth_ = otherDepth;
      width_ = otherWidth;
      totalCount_ = otherTotalCount;
      hashA_.resize(otherDepth);
      for (int32_t i = 0; i < otherDepth; ++i) {
        hashA_[i] = readBigEndianLong(buffer);
      }
      table_.resize(static_cast<size_t>(otherDepth) * otherWidth);
      for (int32_t i = 0; i < otherDepth; ++i) {
        for (int32_t j = 0; j < otherWidth; ++j) {
          table_[static_cast<size_t>(i) * otherWidth + j] =
              readBigEndianLong(buffer);
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
      auto otherHash = readBigEndianLong(buffer);
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
        count = wrappingAdd(count, readBigEndianLong(buffer));
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
  const auto* constant = argument->as<ConstantVector<T>>();
  VELOX_CHECK_NOT_NULL(constant);
  VELOX_CHECK_GT(constant->size(), 0);
  VELOX_USER_CHECK(
      !constant->isNullAt(0), "{} argument must not be null", name);
  return constant->valueAt(0);
}

// Retains literal parameters independently of per-group state so empty groups
// can be serialized before receiving any rows.
class CountMinSketchParameters {
 public:
  // Capture literal parameters before processing input so empty input and
  // fully masked groups can produce valid empty sketches.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    if (constantInputs.size() == 1) {
      // Intermediate input carries the dimensions in the serialized sketch.
      return;
    }
    VELOX_CHECK_EQ(constantInputs.size(), 4);
    VELOX_USER_CHECK_NOT_NULL(
        constantInputs[1], "eps argument must be constant");
    VELOX_USER_CHECK_NOT_NULL(
        constantInputs[2], "confidence argument must be constant");
    VELOX_USER_CHECK_NOT_NULL(
        constantInputs[3], "seed argument must be constant");
    computeDimensions(constantInputs);
  }

  // Initializes a sketch only when input arrives, preserving lazy allocation
  // for fully masked and empty groups.
  void initialize(CountMinSketchAccumulator& accumulator) const {
    accumulator.initialize(depth_, width_, hashA_);
  }

  // Intermediate-only aggregation has no dimensions until a sketch is merged.
  bool writeEmptySketch(exec::out_type<Varbinary>& output) const {
    if (depth_ == 0) {
      return false;
    }
    output.resize(serializedSketchSize(depth_, width_));
    serializeEmptySketch(output.data(), depth_, width_, hashA_);
    return true;
  }

 private:
  void computeDimensions(const std::vector<VectorPtr>& arguments) {
    const auto eps = readConstantArgument<double>(arguments[1], "eps");
    VELOX_USER_CHECK_GT(eps, 0.0, "eps must be positive");

    const auto confidence =
        readConstantArgument<double>(arguments[2], "confidence");
    VELOX_USER_CHECK_GT(confidence, 0.0, "confidence must be positive");
    VELOX_USER_CHECK_LT(confidence, 1.0, "confidence must be less than 1.0");

    int32_t seed{0};
    if (arguments[3]->type()->kind() == TypeKind::INTEGER) {
      seed = readConstantArgument<int32_t>(arguments[3], "seed");
    } else {
      seed = static_cast<int32_t>(
          readConstantArgument<int64_t>(arguments[3], "seed"));
    }

    // Compute depth and width matching Spark's CountMinSketchImpl.
    // Validate the width before narrowing to int32 to avoid undefined behavior
    // when 2 / eps is out of int32 range (very small eps), and division by
    // zero later when eps is +infinity (width 0).
    double widthDouble = std::ceil(2.0 / eps);
    static constexpr char kWidthOutOfRangeMessage[] =
        "count_min_sketch width out of range for eps: {}";
    VELOX_USER_CHECK(std::isfinite(widthDouble), kWidthOutOfRangeMessage, eps);
    VELOX_USER_CHECK_GE(widthDouble, 1.0, kWidthOutOfRangeMessage, eps);
    VELOX_USER_CHECK_LE(
        widthDouble,
        static_cast<double>(std::numeric_limits<int32_t>::max()),
        kWidthOutOfRangeMessage,
        eps);
    width_ = static_cast<int32_t>(widthDouble);
    // Use log1p(-c) to match Spark's exact formula and rounding.
    depth_ = static_cast<int32_t>(
        std::ceil(-std::log1p(-confidence) / std::log(2.0)));
    const auto outputSize = serializedSketchSize(depth_, width_);
    VELOX_USER_CHECK_LE(
        outputSize,
        std::numeric_limits<int32_t>::max(),
        "count_min_sketch serialized size out of range");

    // Initialize hashA using Java-compatible Random.
    JavaRandom random(seed);
    hashA_.resize(depth_);
    for (int32_t i = 0; i < depth_; ++i) {
      hashA_[i] = random.nextInt(std::numeric_limits<int32_t>::max());
    }
  }

  int32_t depth_{0};
  int32_t width_{0};
  std::vector<int64_t> hashA_;
};

// Supplies typed row operations while the adapter owns group lifecycle and
// result-vector management.
template <typename TInput, typename TSeed>
class CountMinSketchAggregate {
 public:
  using InputType = exec::AggregateInputType<TInput, double, double, TSeed>;
  using IntermediateType = Varbinary;
  using OutputType = Varbinary;

  // Empty raw groups return a non-null sketch even without addInput calls.
  static constexpr bool default_null_behavior_ = false;

  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    parameters_.setConstantInputs(constantInputs);
  }

  // Keeps sketch storage in the row allocator and borrows literal parameters
  // from the function object, which outlives its groups.
  class AccumulatorType {
   public:
    static constexpr bool is_fixed_size_ = false;
    static constexpr bool is_aligned_ = true;

    AccumulatorType(
        HashStringAllocator* allocator,
        const CountMinSketchAggregate* function)
        : sketch_(allocator), parameters_(&function->parameters_) {}

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInput> value,
        exec::optional_arg_type<double> /*eps*/,
        exec::optional_arg_type<double> /*confidence*/,
        exec::optional_arg_type<TSeed> /*seed*/) {
      parameters_->initialize(sketch_);
      if (value.has_value()) {
        if constexpr (std::is_integral_v<TInput>) {
          sketch_.addLong(value.value());
        } else {
          const auto bytes = value.value();
          sketch_.addBinary(bytes.data(), bytes.size());
        }
      }
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<IntermediateType> other) {
      if (!other.has_value()) {
        return false;
      }
      sketch_.mergeWith(other.value());
      return true;
    }

    bool writeIntermediateResult(
        bool /*nonNullGroup*/,
        exec::out_type<IntermediateType>& output) const {
      return writeResult(output);
    }

    bool writeFinalResult(
        bool /*nonNullGroup*/,
        exec::out_type<OutputType>& output) const {
      return writeResult(output);
    }

   private:
    // Extraction must not initialize or otherwise mutate group state.
    bool writeResult(exec::out_type<Varbinary>& output) const {
      if (!sketch_.initialized()) {
        return parameters_->writeEmptySketch(output);
      }
      output.resize(sketch_.serializedSize());
      sketch_.serialize(output.data());
      return true;
    }

    CountMinSketchAccumulator sketch_;
    const CountMinSketchParameters* parameters_;
  };

 private:
  CountMinSketchParameters parameters_;
};

template <typename TInput>
std::unique_ptr<exec::Aggregate> createCountMinSketchAggregate(
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& argumentTypes,
    const TypePtr& resultType) {
  if (exec::isRawInput(step) && argumentTypes[3]->kind() == TypeKind::INTEGER) {
    return std::make_unique<
        exec::SimpleAggregateAdapter<CountMinSketchAggregate<TInput, int32_t>>>(
        step, argumentTypes, resultType);
  }
  return std::make_unique<
      exec::SimpleAggregateAdapter<CountMinSketchAggregate<TInput, int64_t>>>(
      step, argumentTypes, resultType);
}

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

  exec::AggregateFunctionMetadata metadata;
  metadata.orderSensitive = false;

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argumentTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        if (!exec::isRawInput(step)) {
          // The serialized state does not depend on the original input type.
          return createCountMinSketchAggregate<int64_t>(
              step, argumentTypes, resultType);
        }
        VELOX_CHECK_EQ(argumentTypes.size(), 4);
        switch (argumentTypes[0]->kind()) {
          case TypeKind::TINYINT:
            return createCountMinSketchAggregate<int8_t>(
                step, argumentTypes, resultType);
          case TypeKind::SMALLINT:
            return createCountMinSketchAggregate<int16_t>(
                step, argumentTypes, resultType);
          case TypeKind::INTEGER:
            return createCountMinSketchAggregate<int32_t>(
                step, argumentTypes, resultType);
          case TypeKind::BIGINT:
            return createCountMinSketchAggregate<int64_t>(
                step, argumentTypes, resultType);
          case TypeKind::VARCHAR:
            return createCountMinSketchAggregate<Varchar>(
                step, argumentTypes, resultType);
          case TypeKind::VARBINARY:
            return createCountMinSketchAggregate<Varbinary>(
                step, argumentTypes, resultType);
          default:
            VELOX_UNREACHABLE(
                "Unsupported type for count_min_sketch: {}",
                argumentTypes[0]->kind());
        }
      },
      metadata,
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
