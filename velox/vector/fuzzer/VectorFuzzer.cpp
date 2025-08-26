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

#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <boost/random/uniform_int_distribution.hpp>
#include <fmt/format.h>
#include <codecvt>
#include <locale>

#include "velox/common/base/Exceptions.h"
#include "velox/common/fuzzer/Utils.h"
#include "velox/functions/prestosql/types/IPPrefixType.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/NullsBuilder.h"
#include "velox/vector/VectorTypeUtils.h"
#include "velox/vector/fuzzer/ConstrainedVectorGenerator.h"
#include "velox/vector/fuzzer/Utils.h"

namespace facebook::velox {

namespace {

using fuzzer::rand;
using fuzzer::randDate;

// Structure to help temporary changes to Options. This objects saves the
// current state of the Options object, and restores it when it's destructed.
// For instance, if you would like to temporarily disable nulls for a particular
// recursive call:
//
//  {
//    ScopedOptions scopedOptions(this);
//    opts_.nullRatio = 0;
//    // perhaps change other opts_ values.
//    vector = fuzzFlat(...);
//  }
//  // At this point, opts_ would have the original values again.
//
struct ScopedOptions {
  explicit ScopedOptions(VectorFuzzer* fuzzer)
      : fuzzer(fuzzer), savedOpts(fuzzer->getOptions()) {}

  ~ScopedOptions() {
    fuzzer->setOptions(savedOpts);
  }

  // Stores a copy of Options so we can restore at destruction time.
  VectorFuzzer* fuzzer;
  VectorFuzzer::Options savedOpts;
};

size_t getElementsVectorLength(
    const VectorFuzzer::Options& opts,
    vector_size_t size) {
  if (!opts.containerVariableLength) {
    VELOX_USER_CHECK_LE(
        size * opts.containerLength,
        opts.complexElementsMaxSize,
        "Requested fixed opts.containerVariableLength can't be satisfied: "
        "increase opts.complexElementsMaxSize, reduce opts.containerLength"
        " or make opts.containerVariableLength=true");
  }
  return std::min(size * opts.containerLength, opts.complexElementsMaxSize);
}

int64_t randShortDecimal(const TypePtr& type, FuzzerGenerator& rng) {
  auto precision = type->asShortDecimal().precision();
  return rand<int64_t>(rng) % DecimalUtil::kPowersOfTen[precision];
}

int128_t randLongDecimal(const TypePtr& type, FuzzerGenerator& rng) {
  auto precision = type->asLongDecimal().precision();
  return rand<int128_t>(rng) % DecimalUtil::kPowersOfTen[precision];
}

/// Generates a random string (string size and encoding are passed through
/// Options). Returns a StringView which uses `buf` as the underlying buffer.
StringView randString(
    FuzzerGenerator& rng,
    const VectorFuzzer::Options& opts,
    std::string& buf,
    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t>& converter) {
  const size_t stringLength = opts.stringVariableLength
      ? rand<uint32_t>(rng) % opts.stringLength
      : opts.stringLength;

  randString(rng, stringLength, opts.charEncodings, buf, converter);
  return StringView(buf);
}

template <TypeKind kind>
VectorPtr fuzzConstantPrimitiveImpl(
    memory::MemoryPool* pool,
    const TypePtr& type,
    vector_size_t size,
    FuzzerGenerator& rng,
    const VectorFuzzer::Options& opts,
    const AbstractInputGeneratorPtr& customGenerator) {
  if (customGenerator) {
    return fuzzer::ConstrainedVectorGenerator::generateConstant(
        customGenerator, size, pool);
  }

  using TCpp = typename TypeTraits<kind>::NativeType;
  if constexpr (std::is_same_v<TCpp, StringView>) {
    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;
    std::string buf;
    auto stringView = randString(rng, opts, buf, converter);

    return std::make_shared<ConstantVector<TCpp>>(
        pool, size, false, type, std::move(stringView));
  }
  if constexpr (std::is_same_v<TCpp, Timestamp>) {
    return std::make_shared<ConstantVector<TCpp>>(
        pool, size, false, type, randTimestamp(rng, opts.timestampPrecision));
  } else if (type->isDate()) {
    return std::make_shared<ConstantVector<int32_t>>(
        pool, size, false, type, randDate(rng));
  } else if (type->isShortDecimal()) {
    return std::make_shared<ConstantVector<int64_t>>(
        pool, size, false, type, randShortDecimal(type, rng));
  } else if (type->isLongDecimal()) {
    return std::make_shared<ConstantVector<int128_t>>(
        pool, size, false, type, randLongDecimal(type, rng));
  } else {
    return std::make_shared<ConstantVector<TCpp>>(
        pool, size, false, type, rand<TCpp>(rng, opts.dataSpec));
  }
}

template <TypeKind kind>
void fuzzFlatPrimitiveImpl(
    const VectorPtr& vector,
    FuzzerGenerator& rng,
    const VectorFuzzer::Options& opts) {
  using TFlat = typename KindToFlatVector<kind>::type;
  using TCpp = typename TypeTraits<kind>::NativeType;

  auto flatVector = vector->as<TFlat>();
  std::string strBuf;

  std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;
  for (size_t i = 0; i < vector->size(); ++i) {
    if constexpr (std::is_same_v<TCpp, StringView>) {
      flatVector->set(i, randString(rng, opts, strBuf, converter));
    } else if constexpr (std::is_same_v<TCpp, Timestamp>) {
      flatVector->set(i, randTimestamp(rng, opts.timestampPrecision));
    } else if constexpr (std::is_same_v<TCpp, int64_t>) {
      if (vector->type()->isIntervalDayTime()) {
        flatVector->set(
            i,
            rand<TCpp>(
                rng,
                -VectorFuzzer::kMaxAllowedIntervalDayTime,
                VectorFuzzer::kMaxAllowedIntervalDayTime));
      } else if (vector->type()->isShortDecimal()) {
        flatVector->set(i, randShortDecimal(vector->type(), rng));
      } else {
        flatVector->set(i, rand<TCpp>(rng, opts.dataSpec));
      }
    } else if constexpr (std::is_same_v<TCpp, int128_t>) {
      if (vector->type()->isLongDecimal()) {
        flatVector->set(i, randLongDecimal(vector->type(), rng));
      } else if (vector->type()->isHugeint()) {
        flatVector->set(i, rand<int128_t>(rng, opts.dataSpec));
      } else {
        VELOX_NYI();
      }
    } else if constexpr (std::is_same_v<TCpp, int32_t>) {
      if (vector->type()->isIntervalYearMonth()) {
        flatVector->set(
            i,
            rand<TCpp>(
                rng,
                -VectorFuzzer::kMaxAllowedIntervalYearMonth,
                VectorFuzzer::kMaxAllowedIntervalYearMonth));
      } else if (vector->type()->isDate()) {
        flatVector->set(i, randDate(rng));
      } else {
        flatVector->set(i, rand<TCpp>(rng));
      }
    } else {
      flatVector->set(i, rand<TCpp>(rng, opts.dataSpec));
    }
  }
}

// Servers as a wrapper around a vector that will be used to load a lazyVector.
// Ensures that the loaded vector will only contain valid rows for the row set
// that it was loaded for. NOTE: If the vector is a multi-level dictionary, the
// indices from all the dictionaries are combined.
class VectorLoaderWrap : public VectorLoader {
 public:
  explicit VectorLoaderWrap(VectorPtr vector) : vector_(vector) {}

  void loadInternal(
      RowSet rowSet,
      ValueHook* hook,
      vector_size_t resultSize,
      VectorPtr* result) override {
    velox::common::testutil::TestValue::adjust(
        "facebook::velox::{}::VectorLoaderWrap::loadInternal", this);
    VELOX_CHECK(!hook, "VectorLoaderWrap doesn't support ValueHook");
    SelectivityVector rows(rowSet.back() + 1, false);
    for (auto row : rowSet) {
      rows.setValid(row, true);
    }
    rows.updateBounds();
    *result = makeEncodingPreservedCopy(rows, resultSize);
  }

 private:
  // Returns a copy of 'vector_' while retaining dictionary encoding if present.
  // Multiple dictionary layers are collapsed into one.
  VectorPtr makeEncodingPreservedCopy(
      SelectivityVector& rows,
      vector_size_t vectorSize);
  VectorPtr vector_;
};

bool hasNestedDictionaryLayers(const VectorPtr& baseVector) {
  return baseVector && VectorEncoding::isDictionary(baseVector->encoding()) &&
      VectorEncoding::isDictionary(baseVector->valueVector()->encoding());
}

// Returns an AbstractInputGeneratorPtr for the given type if it's a custom
// type, otherwise returns null.
AbstractInputGeneratorPtr maybeGetCustomTypeInputGenerator(
    const TypePtr& type,
    const double nullRatio,
    FuzzerGenerator& rng,
    memory::MemoryPool* pool) {
  if (customTypeExists(type->name())) {
    InputGeneratorConfig config{rand<uint32_t>(rng), nullRatio, pool, type};
    return getCustomTypeInputGenerator(type->name(), config);
  }

  return nullptr;
}
} // namespace

VectorPtr VectorFuzzer::fuzzNotNull(
    const TypePtr& type,
    const AbstractInputGeneratorPtr& customGenerator) {
  return fuzzNotNull(type, opts_.vectorSize, customGenerator);
}

VectorPtr VectorFuzzer::fuzzNotNull(
    const TypePtr& type,
    vector_size_t size,
    const AbstractInputGeneratorPtr& customGenerator) {
  ScopedOptions restorer(this);
  opts_.nullRatio = 0;
  return fuzz(type, size, customGenerator);
}

VectorPtr VectorFuzzer::fuzz(
    const TypePtr& type,
    const AbstractInputGeneratorPtr& customGenerator) {
  return fuzz(type, opts_.vectorSize, customGenerator);
}

VectorPtr VectorFuzzer::fuzz(
    const TypePtr& type,
    vector_size_t size,
    const AbstractInputGeneratorPtr& customGenerator) {
  VectorPtr vector;
  vector_size_t vectorSize = size;
  const auto inputGenerator = customGenerator
      ? customGenerator
      : maybeGetCustomTypeInputGenerator(type, opts_.nullRatio, rng_, pool_);

  bool usingLazyVector = opts_.allowLazyVector && coinToss(0.1);
  // Lazy Vectors cannot be sliced, so we skip this if using lazy wrapping.
  if (opts_.allowSlice && !usingLazyVector && coinToss(0.1)) {
    // Extend the underlying vector to allow slicing later.
    vectorSize += rand<uint32_t>(rng_) % 8;
  }

  // 20% chance of adding a constant vector.
  if (opts_.allowConstantVector && coinToss(0.2)) {
    vector = fuzzConstant(type, vectorSize, inputGenerator);
  } else if (type->isOpaque()) {
    vector = fuzzFlatOpaque(type, vectorSize);
  } else if (inputGenerator) {
    vector = fuzzFlat(type, vectorSize, inputGenerator);
  } else {
    vector = type->isPrimitiveType() ? fuzzFlatPrimitive(type, vectorSize)
                                     : fuzzComplex(type, vectorSize);
  }

  if (vectorSize > size) {
    auto offset = rand<uint32_t>(rng_) % (vectorSize - size + 1);
    vector = vector->slice(offset, size);
  }

  if (usingLazyVector) {
    vector = wrapInLazyVector(vector);
  }

  // Toss a coin and add dictionary indirections.
  while (opts_.allowDictionaryVector && coinToss(0.5)) {
    vectorSize = size;
    if (opts_.allowSlice && !usingLazyVector && vectorSize > 0 &&
        coinToss(0.05)) {
      vectorSize += rand<uint32_t>(rng_) % 8;
    }
    vector = fuzzDictionary(vector, vectorSize);
    if (vectorSize > size) {
      auto offset = rand<uint32_t>(rng_) % (vectorSize - size + 1);
      vector = vector->slice(offset, size);
    }
  }
  VELOX_CHECK_EQ(vector->size(), size);
  return vector;
}

VectorPtr VectorFuzzer::fuzz(const GeneratorSpec& generatorSpec) {
  return generatorSpec.generateData(rng_, pool_, opts_.vectorSize);
}

VectorPtr VectorFuzzer::fuzzConstant(
    const TypePtr& type,
    const AbstractInputGeneratorPtr& customGenerator) {
  return fuzzConstant(type, opts_.vectorSize, customGenerator);
}

VectorPtr VectorFuzzer::fuzzConstant(
    const TypePtr& type,
    vector_size_t size,
    const AbstractInputGeneratorPtr& customGenerator) {
  const auto inputGenerator = customGenerator
      ? customGenerator
      : maybeGetCustomTypeInputGenerator(type, opts_.nullRatio, rng_, pool_);

  // For constants, there are two possible cases:
  // - generate a regular constant vector (only for primitive types).
  // - generate a random vector and wrap it using a constant vector.
  if (type->isPrimitiveType() && coinToss(0.5)) {
    // For regular constant vectors, toss a coin to determine its nullability.
    if (coinToss(opts_.nullRatio)) {
      return BaseVector::createNullConstant(type, size, pool_);
    }
    if (type->isUnKnown()) {
      return BaseVector::createNullConstant(type, size, pool_);
    } else {
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          fuzzConstantPrimitiveImpl,
          type->kind(),
          pool_,
          type,
          size,
          rng_,
          opts_,
          inputGenerator);
    }
  }

  // Otherwise, create constant by wrapping around another vector. This will
  // return a null constant if the element being wrapped is null in the
  // generated inner vector.

  // Inner vector size can't be zero.
  auto innerVectorSize = rand<uint32_t>(rng_) % opts_.vectorSize + 1;
  auto constantIndex = rand<vector_size_t>(rng_) % innerVectorSize;

  ScopedOptions restorer(this);
  opts_.allowLazyVector = false;
  // Have a lower cap on repeated sizes inside constant. Otherwise will OOM when
  // flattening.
  if (opts_.maxConstantContainerSize.has_value()) {
    opts_.containerLength = std::min<int32_t>(
        opts_.maxConstantContainerSize.value(), opts_.containerLength);
    opts_.complexElementsMaxSize = std::min<int32_t>(
        opts_.maxConstantContainerSize.value(), opts_.complexElementsMaxSize);
  }
  return BaseVector::wrapInConstant(
      size, constantIndex, fuzz(type, innerVectorSize, inputGenerator));
}

VectorPtr VectorFuzzer::fuzzFlat(
    const TypePtr& type,
    const AbstractInputGeneratorPtr& customGenerator) {
  return fuzzFlat(type, opts_.vectorSize, customGenerator);
}

VectorPtr VectorFuzzer::fuzzFlatNotNull(const TypePtr& type) {
  return fuzzFlatNotNull(type, opts_.vectorSize);
}

VectorPtr VectorFuzzer::fuzzFlatNotNull(
    const TypePtr& type,
    vector_size_t size) {
  ScopedOptions restorer(this);
  opts_.nullRatio = 0;
  return fuzzFlat(type, size);
}

VectorPtr VectorFuzzer::fuzzFlat(
    const TypePtr& type,
    vector_size_t size,
    const AbstractInputGeneratorPtr& customGenerator) {
  const auto inputGenerator = customGenerator
      ? customGenerator
      : maybeGetCustomTypeInputGenerator(type, opts_.nullRatio, rng_, pool_);
  if (inputGenerator) {
    return fuzzer::ConstrainedVectorGenerator::generateFlat(
        inputGenerator, size, pool_);
  }

  // Primitive types.
  if (type->isPrimitiveType()) {
    return fuzzFlatPrimitive(type, size);
  }
  // Arrays.
  else if (type->isArray()) {
    const auto& elementType = type->asArray().elementType();
    auto elementsLength = getElementsVectorLength(opts_, size);

    auto elements = opts_.containerHasNulls
        ? fuzzFlat(elementType, elementsLength)
        : fuzzFlatNotNull(elementType, elementsLength);
    return fuzzArray(elements, size);
  }
  // Maps.
  else if (type->isMap()) {
    // Do not initialize keys and values inline in the fuzzMap call as C++ does
    // not specify the order they'll be called in, leading to inconsistent
    // results across platforms.
    return fuzzMap(type->asMap().keyType(), type->asMap().valueType(), size);
  }
  // Rows.
  else if (type->isRow()) {
    const auto& rowType = type->asRow();
    std::vector<VectorPtr> childrenVectors;
    childrenVectors.reserve(rowType.children().size());

    for (const auto& childType : rowType.children()) {
      childrenVectors.emplace_back(
          opts_.containerHasNulls ? fuzzFlat(childType, size)
                                  : fuzzFlatNotNull(childType, size));
    }

    return fuzzRow(std::move(childrenVectors), rowType.names(), size);
  } else if (type->isOpaque()) {
    return fuzzFlatOpaque(type, size);
  } else {
    VELOX_UNREACHABLE();
  }
}

VectorPtr VectorFuzzer::fuzzMap(
    const TypePtr& keyType,
    const TypePtr& valueType,
    vector_size_t size) {
  auto length = getElementsVectorLength(opts_, size);

  auto keys = opts_.normalizeMapKeys || !opts_.containerHasNulls
      ? fuzzFlatNotNull(keyType, length)
      : fuzzFlat(keyType, length);
  auto values = opts_.containerHasNulls ? fuzzFlat(valueType, length)
                                        : fuzzFlatNotNull(valueType, length);
  return fuzzMap(keys, values, size);
}

VectorPtr VectorFuzzer::fuzzFlatPrimitive(
    const TypePtr& type,
    vector_size_t size) {
  VELOX_CHECK(type->isPrimitiveType());
  auto vector = BaseVector::create(type, size, pool_);

  if (type->isUnKnown()) {
    auto* rawNulls = vector->mutableRawNulls();
    bits::fillBits(rawNulls, 0, size, bits::kNull);
  } else {
    // First, fill it with random values.
    // TODO: We should bias towards edge cases (min, max, Nan, etc).
    auto kind = vector->typeKind();
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
        fuzzFlatPrimitiveImpl, kind, vector, rng_, opts_);

    // Second, generate a random null vector.
    for (size_t i = 0; i < vector->size(); ++i) {
      if (coinToss(opts_.nullRatio)) {
        vector->setNull(i, true);
      }
    }
  }
  return vector;
}

VectorPtr VectorFuzzer::fuzzComplex(const TypePtr& type, vector_size_t size) {
  ScopedOptions restorer(this);
  opts_.allowLazyVector = false;

  switch (type->kind()) {
    case TypeKind::ROW: {
      if (isIPPrefixType(type)) {
        ScopedOptions restorer(this);
        opts_.containerHasNulls = false;
        return fuzzRow(std::dynamic_pointer_cast<const RowType>(type), size);
      }
      return fuzzRow(std::dynamic_pointer_cast<const RowType>(type), size);
    }

    case TypeKind::ARRAY: {
      const auto& elementType = type->asArray().elementType();
      auto elementsLength = getElementsVectorLength(opts_, size);

      auto elements = opts_.containerHasNulls
          ? fuzz(elementType, elementsLength)
          : fuzzNotNull(elementType, elementsLength);
      return fuzzArray(elements, size);
    }

    case TypeKind::MAP: {
      // Do not initialize keys and values inline in the fuzzMap call as C++
      // does not specify the order they'll be called in, leading to
      // inconsistent results across platforms.
      const auto& keyType = type->asMap().keyType();
      const auto& valueType = type->asMap().valueType();
      auto length = getElementsVectorLength(opts_, size);

      auto keys = opts_.normalizeMapKeys || !opts_.containerHasNulls
          ? fuzzNotNull(keyType, length)
          : fuzz(keyType, length);
      auto values = opts_.containerHasNulls ? fuzz(valueType, length)
                                            : fuzzNotNull(valueType, length);
      return fuzzMap(keys, values, size);
    }

    default:
      VELOX_UNREACHABLE("Unexpected type: {}", type->toString());
  }
  return nullptr; // no-op.
}

VectorPtr VectorFuzzer::fuzzFlatOpaque(
    const TypePtr& type,
    vector_size_t size) {
  VELOX_CHECK(type->isOpaque());
  auto vector = BaseVector::create(type, size, pool_);
  using TFlat = typename KindToFlatVector<TypeKind::OPAQUE>::type;

  auto& opaqueType = type->asOpaque();
  auto flatVector = vector->as<TFlat>();
  auto it = opaqueTypeGenerators_.find(opaqueType.typeIndex());
  VELOX_CHECK(
      it != opaqueTypeGenerators_.end(),
      "generator does not exist for type index. Did you call registerOpaqueTypeGenerator()?");
  auto& opaqueTypeGenerator = it->second;
  for (vector_size_t i = 0; i < vector->size(); ++i) {
    if (coinToss(opts_.nullRatio)) {
      flatVector->setNull(i, true);
    } else {
      flatVector->set(i, opaqueTypeGenerator(rng_));
    }
  }
  return vector;
}

VectorPtr VectorFuzzer::fuzzDictionary(const VectorPtr& vector) {
  return fuzzDictionary(vector, vector->size());
}

VectorPtr VectorFuzzer::fuzzDictionary(
    const VectorPtr& vector,
    vector_size_t size) {
  const size_t vectorSize = vector->size();
  VELOX_CHECK(
      vectorSize > 0 || size == 0,
      "Cannot build a non-empty dictionary on an empty underlying vector");
  BufferPtr indices = fuzzIndices(size, vectorSize);

  auto nulls = opts_.dictionaryHasNulls ? fuzzNulls(size) : nullptr;
  return BaseVector::wrapInDictionary(nulls, indices, size, vector);
}

void VectorFuzzer::fuzzOffsetsAndSizes(
    BufferPtr& offsets,
    BufferPtr& sizes,
    size_t elementsSize,
    size_t size,
    bool isMapType) {
  offsets = allocateOffsets(size, pool_);
  sizes = allocateSizes(size, pool_);
  auto rawOffsets = offsets->asMutable<vector_size_t>();
  auto rawSizes = sizes->asMutable<vector_size_t>();

  size_t containerAvgLength = std::max(elementsSize / size, 1UL);
  size_t childSize = 0;
  size_t length = 0;

  for (auto i = 0; i < size; ++i) {
    rawOffsets[i] = childSize;

    // If variable length, generate a random number between zero and 2 *
    // containerAvgLength (so that the average of generated containers size is
    // equal to number of input elements).
    if (opts_.containerVariableLength) {
      length = rand<uint32_t>(rng_) % (containerAvgLength * 2);
    } else {
      length = containerAvgLength;
    }

    // If we exhausted the available elements, adjust length to fit remaining
    // elements.
    if ((childSize + length) > elementsSize) {
      length = elementsSize > childSize ? elementsSize - childSize : 0;
    }
    rawSizes[i] = length;
    childSize += length;
  }
}

ArrayVectorPtr VectorFuzzer::fuzzArray(
    const TypePtr& elementType,
    vector_size_t size) {
  const auto length = getElementsVectorLength(opts_, size);

  auto elements = opts_.containerHasNulls
      ? fuzzFlat(elementType, length)
      : fuzzFlatNotNull(elementType, length);
  return fuzzArray(elements, size);
}

ArrayVectorPtr VectorFuzzer::fuzzArray(
    const VectorPtr& elements,
    vector_size_t size) {
  BufferPtr offsets, sizes;
  fuzzOffsetsAndSizes(offsets, sizes, elements->size(), size);
  return std::make_shared<ArrayVector>(
      pool_,
      ARRAY(elements->type()),
      fuzzNulls(size),
      size,
      offsets,
      sizes,
      elements);
}

VectorPtr VectorFuzzer::normalizeMapKeys(
    const VectorPtr& keys,
    size_t mapSize,
    BufferPtr& offsets,
    BufferPtr& sizes) {
  // Map keys cannot be null.
  const auto& nulls = keys->nulls();
  if (nulls) {
    VELOX_CHECK_EQ(
        BaseVector::countNulls(nulls, 0, keys->size()),
        0,
        "Map keys cannot be null when opt.normalizeMapKeys is true");
  }

  auto rawOffsets = offsets->as<vector_size_t>();
  auto rawSizes = sizes->asMutable<vector_size_t>();

  // Looks for duplicate key values.
  std::unordered_set<uint64_t> set;
  for (size_t i = 0; i < mapSize; ++i) {
    set.clear();

    for (size_t j = 0; j < rawSizes[i]; ++j) {
      vector_size_t idx = rawOffsets[i] + j;
      uint64_t hash = keys->hashValueAt(idx);

      // If we find the same hash (either same key value or hash colision), we
      // cut it short by reducing this element's map size. This should not
      // happen frequently.
      auto it = set.find(hash);
      if (it != set.end()) {
        rawSizes[i] = j;
        break;
      }
      set.insert(hash);
    }
  }
  return keys;
}

MapVectorPtr VectorFuzzer::ensureUniqueMapContent(
    const VectorPtr& keys,
    const VectorPtr& values,
    vector_size_t size) {
  // Use the original key and value types to maintain schema compatibility
  TypePtr keyType = keys->type();
  TypePtr valueType = values->type();

  // Check if this map type has unsupported key/value types for unique content
  // generation Boolean types only have 2 possible values, insufficient for
  // generating unique content across many maps, so fall back to regular map
  // generation
  if (keyType->isBoolean() || valueType->isBoolean()) {
    // Generate regular map with the provided keys and values without unique
    // content
    size_t elementsSize = std::min(keys->size(), values->size());
    BufferPtr offsets, sizes;
    fuzzOffsetsAndSizes(offsets, sizes, elementsSize, size);
    return std::make_shared<MapVector>(
        pool_,
        MAP(keyType, valueType),
        fuzzNulls(size),
        size,
        offsets,
        sizes,
        opts_.normalizeMapKeys ? normalizeMapKeys(keys, size, offsets, sizes)
                               : keys,
        values);
  }

  // Generate completely unique map content across all map instances
  // This includes: global key uniqueness, global value uniqueness,
  // global key-value pair uniqueness, and no duplicates within each map
  std::vector<VectorPtr> allKeyElements;
  std::vector<VectorPtr> allValueElements;
  std::vector<vector_size_t> newOffsets(size);
  std::vector<vector_size_t> newSizes(size);

  // Global tracking to ensure complete uniqueness across ALL maps
  std::unordered_set<uint64_t> globalSeenKeyHashes;
  std::unordered_set<uint64_t> globalSeenValueHashes;
  std::unordered_set<std::string> globalSeenKeyValuePairs;

  vector_size_t currentOffset = 0;

  for (vector_size_t mapIdx = 0; mapIdx < size; ++mapIdx) {
    newOffsets[mapIdx] = currentOffset;

    // Generate varied map sizes to ensure different maps have different
    // structures Use prime numbers and different formulas to maximize variation
    vector_size_t mapSize;
    if (opts_.containerLength <= 1) {
      mapSize = 1;
    } else {
      // Use different formulas for different maps to ensure variety
      switch (mapIdx % 4) {
        case 0:
          mapSize = 1 + (mapIdx * 7) % opts_.containerLength;
          break;
        case 1:
          mapSize = 1 + (mapIdx * 11 + 3) % opts_.containerLength;
          break;
        case 2:
          mapSize = 1 + (mapIdx * 13 + 5) % opts_.containerLength;
          break;
        default:
          mapSize = 1 + (mapIdx * 17 + 7) % opts_.containerLength;
          break;
      }
      mapSize = std::max(
          static_cast<vector_size_t>(1),
          std::min(mapSize, static_cast<vector_size_t>(opts_.containerLength)));
    }

    // Generate unique key-value pairs for this map that are globally unique
    std::vector<VectorPtr> mapKeyElements;
    std::vector<VectorPtr> mapValueElements;

    for (vector_size_t elemIdx = 0; elemIdx < mapSize; ++elemIdx) {
      vector_size_t attempts = 0;
      const vector_size_t maxAttempts = 10000;
      bool foundUniqueContent = false;

      while (!foundUniqueContent && attempts < maxAttempts) {
        attempts++;

        // Create highly varied seeds using multiple prime numbers and offsets
        // to ensure maximum diversity across all maps and elements
        auto baseSeed = rand<uint32_t>(rng_);
        auto tempSeed = baseSeed + mapIdx * 1000003UL + // Large prime
            elemIdx * 1000033UL + // Different large prime
            attempts * 1000037UL + // Another large prime
            size * 1000039UL + // Yet another large prime
            currentOffset * 1000081UL + // More variation
            (mapIdx * elemIdx) * 1000099UL + // Cross product for more variety
            (mapIdx + elemIdx + attempts) * 1000117UL; // Additional entropy

        // Create a temporary fuzzer with the unique seed
        VectorFuzzer tempFuzzer(opts_, pool_, tempSeed);

        // Generate a single key-value pair
        auto singleKey = opts_.normalizeMapKeys || !opts_.containerHasNulls
            ? tempFuzzer.fuzzFlatNotNull(keyType, 1)
            : tempFuzzer.fuzzFlat(keyType, 1);
        auto singleValue = opts_.containerHasNulls
            ? tempFuzzer.fuzzFlat(valueType, 1)
            : tempFuzzer.fuzzFlatNotNull(valueType, 1);

        // Check if this key-value pair is globally unique
        uint64_t keyHash = singleKey->hashValueAt(0);
        uint64_t valueHash = singleValue->hashValueAt(0);

        // Create a combined hash for the key-value pair
        std::string keyValuePair =
            std::to_string(keyHash) + "_" + std::to_string(valueHash);

        if (globalSeenKeyHashes.find(keyHash) == globalSeenKeyHashes.end() &&
            globalSeenValueHashes.find(valueHash) ==
                globalSeenValueHashes.end() &&
            globalSeenKeyValuePairs.find(keyValuePair) ==
                globalSeenKeyValuePairs.end()) {
          globalSeenKeyHashes.insert(keyHash);
          globalSeenValueHashes.insert(valueHash);
          globalSeenKeyValuePairs.insert(keyValuePair);

          mapKeyElements.push_back(singleKey);
          mapValueElements.push_back(singleValue);
          foundUniqueContent = true;
        }
      }

      // If we couldn't find globally unique content, force uniqueness using
      // deterministic approach
      if (!foundUniqueContent) {
        // Create completely unique deterministic content based on global
        // position
        uint64_t globalElementIndex = currentOffset + elemIdx;

        // Generate deterministic unique key
        if (keyType->isVarchar()) {
          std::string uniqueKey = "globally_unique_key_" +
              std::to_string(globalElementIndex) + "_map_" +
              std::to_string(mapIdx) + "_elem_" + std::to_string(elemIdx) +
              "_attempt_" + std::to_string(attempts);

          auto keyBuffer =
              AlignedBuffer::allocate<char>(uniqueKey.size(), pool_);
          std::memcpy(
              keyBuffer->asMutable<char>(), uniqueKey.data(), uniqueKey.size());

          auto keyVector = std::make_shared<FlatVector<StringView>>(
              pool_,
              VARCHAR(),
              nullptr,
              1,
              nullptr,
              std::vector<BufferPtr>{keyBuffer});
          keyVector->set(
              0, StringView(keyBuffer->as<char>(), uniqueKey.size()));

          mapKeyElements.push_back(keyVector);
        } else if (keyType->isInteger()) {
          // Use global element index to ensure uniqueness across all maps
          int32_t uniqueKeyValue = static_cast<int32_t>(
              globalElementIndex * 1000000 + mapIdx * 1000 + elemIdx);
          auto keyVector = BaseVector::create(INTEGER(), 1, pool_);
          keyVector->asFlatVector<int32_t>()->set(0, uniqueKeyValue);
          mapKeyElements.push_back(keyVector);
        } else if (keyType->isBigint()) {
          // Use global element index to ensure uniqueness across all maps
          int64_t uniqueKeyValue =
              static_cast<int64_t>(globalElementIndex) * 1000000000000LL +
              static_cast<int64_t>(mapIdx) * 1000000LL + elemIdx;
          auto keyVector = BaseVector::create(BIGINT(), 1, pool_);
          keyVector->asFlatVector<int64_t>()->set(0, uniqueKeyValue);
          mapKeyElements.push_back(keyVector);
        } else {
          // For other key types, use a globally unique seed
          auto tempSeed = globalElementIndex * 2000000003UL +
              mapIdx * 2000000011UL + elemIdx * 2000000017UL +
              attempts * 2000000023UL;
          VectorFuzzer tempFuzzer(opts_, pool_, tempSeed);
          auto singleKey = opts_.normalizeMapKeys || !opts_.containerHasNulls
              ? tempFuzzer.fuzzFlatNotNull(keyType, 1)
              : tempFuzzer.fuzzFlat(keyType, 1);
          mapKeyElements.push_back(singleKey);
        }

        // Generate corresponding globally unique value
        if (valueType->isVarchar()) {
          std::string uniqueValue = "globally_unique_value_" +
              std::to_string(globalElementIndex) + "_map_" +
              std::to_string(mapIdx) + "_elem_" + std::to_string(elemIdx) +
              "_attempt_" + std::to_string(attempts);

          auto valueBuffer =
              AlignedBuffer::allocate<char>(uniqueValue.size(), pool_);
          std::memcpy(
              valueBuffer->asMutable<char>(),
              uniqueValue.data(),
              uniqueValue.size());

          auto valueVector = std::make_shared<FlatVector<StringView>>(
              pool_,
              VARCHAR(),
              nullptr,
              1,
              nullptr,
              std::vector<BufferPtr>{valueBuffer});
          valueVector->set(
              0, StringView(valueBuffer->as<char>(), uniqueValue.size()));

          mapValueElements.push_back(valueVector);
        } else if (valueType->isRow()) {
          // For ROW types, generate deterministic unique content for each field
          const auto& rowType = valueType->asRow();
          std::vector<VectorPtr> childVectors;
          childVectors.reserve(rowType.size());

          for (auto fieldIdx = 0; fieldIdx < rowType.size(); ++fieldIdx) {
            const auto& fieldType = rowType.childAt(fieldIdx);
            auto fieldSeed = globalElementIndex * 4000000007UL +
                mapIdx * 4000000009UL + elemIdx * 4000000013UL +
                fieldIdx * 4000000021UL + attempts * 4000000037UL;

            VectorFuzzer fieldFuzzer(opts_, pool_, fieldSeed);
            auto fieldVector = opts_.containerHasNulls
                ? fieldFuzzer.fuzzFlat(fieldType, 1)
                : fieldFuzzer.fuzzFlatNotNull(fieldType, 1);
            childVectors.push_back(fieldVector);
          }

          auto rowVector = std::make_shared<RowVector>(
              pool_, valueType, nullptr, 1, std::move(childVectors));
          mapValueElements.push_back(rowVector);
        } else {
          // For other value types, use a globally unique seed
          auto tempSeed = globalElementIndex * 3000000007UL +
              mapIdx * 3000000013UL + elemIdx * 3000000019UL +
              attempts * 3000000029UL;
          VectorFuzzer tempFuzzer(opts_, pool_, tempSeed);
          auto singleValue = opts_.containerHasNulls
              ? tempFuzzer.fuzzFlat(valueType, 1)
              : tempFuzzer.fuzzFlatNotNull(valueType, 1);
          mapValueElements.push_back(singleValue);
        }
      }
    }

    // Add all elements for this map
    for (size_t i = 0; i < mapKeyElements.size(); ++i) {
      allKeyElements.push_back(mapKeyElements[i]);
      allValueElements.push_back(mapValueElements[i]);
    }

    newSizes[mapIdx] = mapKeyElements.size();
    currentOffset += mapKeyElements.size();
  }

  // Concatenate all key and value elements
  VectorPtr newKeys;
  VectorPtr newValues;

  if (allKeyElements.empty()) {
    newKeys = BaseVector::create(keyType, 0, pool_);
    newValues = BaseVector::create(valueType, 0, pool_);
  } else {
    // Calculate total size for keys and values
    vector_size_t totalKeySize = 0;
    for (const auto& keyElement : allKeyElements) {
      totalKeySize += keyElement->size();
    }

    // Create new vectors with the total size
    newKeys = BaseVector::create(keyType, totalKeySize, pool_);
    newValues = BaseVector::create(valueType, totalKeySize, pool_);

    // Copy data from each element
    vector_size_t copyOffset = 0;
    for (size_t i = 0; i < allKeyElements.size(); ++i) {
      const auto& keyElement = allKeyElements[i];
      const auto& valueElement = allValueElements[i];

      newKeys->copy(keyElement.get(), copyOffset, 0, keyElement->size());
      newValues->copy(valueElement.get(), copyOffset, 0, valueElement->size());

      copyOffset += keyElement->size();
    }
  }

  // Create new offset and size buffers
  auto newOffsetsBuffer = AlignedBuffer::allocate<vector_size_t>(size, pool_);
  auto newSizesBuffer = AlignedBuffer::allocate<vector_size_t>(size, pool_);

  auto rawNewOffsets = newOffsetsBuffer->asMutable<vector_size_t>();
  auto rawNewSizes = newSizesBuffer->asMutable<vector_size_t>();

  for (vector_size_t i = 0; i < size; ++i) {
    rawNewOffsets[i] = newOffsets[i];
    rawNewSizes[i] = newSizes[i];
  }

  return std::make_shared<MapVector>(
      pool_,
      MAP(keyType, valueType),
      fuzzNulls(size),
      size,
      newOffsetsBuffer,
      newSizesBuffer,
      newKeys,
      newValues);
}

MapVectorPtr VectorFuzzer::fuzzMap(
    const VectorPtr& keys,
    const VectorPtr& values,
    vector_size_t size) {
  // If ensureUniqueMapContent is enabled, use it for complete uniqueness
  // across all maps (keys, values, and key-value pairs)
  if (opts_.ensureUniqueMapContent) {
    return ensureUniqueMapContent(keys, values, size);
  }

  // Default map generation with possible duplicate keys and values
  size_t elementsSize = std::min(keys->size(), values->size());
  BufferPtr offsets, sizes;
  fuzzOffsetsAndSizes(offsets, sizes, elementsSize, size);
  return std::make_shared<MapVector>(
      pool_,
      MAP(keys->type(), values->type()),
      fuzzNulls(size),
      size,
      offsets,
      sizes,
      opts_.normalizeMapKeys ? normalizeMapKeys(keys, size, offsets, sizes)
                             : keys,
      values);
}

RowVectorPtr VectorFuzzer::fuzzInputRow(
    const RowTypePtr& rowType,
    const std::vector<AbstractInputGeneratorPtr>& inputGenerators) {
  return fuzzRow(rowType, opts_.vectorSize, false, inputGenerators);
}

RowVectorPtr VectorFuzzer::fuzzInputFlatRow(const RowTypePtr& rowType) {
  std::vector<VectorPtr> children;
  auto size = static_cast<vector_size_t>(opts_.vectorSize);
  children.reserve(rowType->size());
  for (auto i = 0; i < rowType->size(); ++i) {
    children.emplace_back(fuzzFlat(rowType->childAt(i), size));
  }

  return std::make_shared<RowVector>(
      pool_, rowType, nullptr, size, std::move(children));
}

RowVectorPtr VectorFuzzer::fuzzRow(
    std::vector<VectorPtr>&& children,
    std::vector<std::string> childrenNames,
    vector_size_t size) {
  std::vector<TypePtr> types;
  types.reserve(children.size());

  for (const auto& child : children) {
    types.emplace_back(child->type());
  }

  return std::make_shared<RowVector>(
      pool_,
      ROW(std::move(childrenNames), std::move(types)),
      fuzzNulls(size),
      size,
      std::move(children));
}

RowVectorPtr VectorFuzzer::fuzzRow(
    std::vector<VectorPtr>&& children,
    vector_size_t size) {
  std::vector<TypePtr> types;
  types.reserve(children.size());

  for (const auto& child : children) {
    types.emplace_back(child->type());
  }

  return std::make_shared<RowVector>(
      pool_, ROW(std::move(types)), fuzzNulls(size), size, std::move(children));
}

RowVectorPtr VectorFuzzer::fuzzRow(const RowTypePtr& rowType) {
  ScopedOptions restorer(this);
  opts_.allowLazyVector = false;
  return fuzzRow(rowType, opts_.vectorSize);
}

RowVectorPtr VectorFuzzer::fuzzRow(
    const RowTypePtr& rowType,
    vector_size_t size,
    bool allowTopLevelNulls,
    const std::vector<AbstractInputGeneratorPtr>& inputGenerators) {
  std::vector<VectorPtr> children;
  children.reserve(rowType->size());

  const AbstractInputGeneratorPtr kNoInputGenerator{nullptr};
  for (auto i = 0; i < rowType->size(); ++i) {
    const auto& inputGenerator =
        inputGenerators.size() > i ? inputGenerators[i] : kNoInputGenerator;
    children.push_back(
        opts_.containerHasNulls
            ? fuzz(rowType->childAt(i), size, inputGenerator)
            : fuzzNotNull(rowType->childAt(i), size, inputGenerator));
  }

  return std::make_shared<RowVector>(
      pool_,
      rowType,
      allowTopLevelNulls ? fuzzNulls(size) : nullptr,
      size,
      std::move(children));
}

BufferPtr VectorFuzzer::fuzzNulls(vector_size_t size) {
  NullsBuilder builder{size, pool_};
  for (size_t i = 0; i < size; ++i) {
    if (coinToss(opts_.nullRatio)) {
      builder.setNull(i);
    }
  }
  return builder.build();
}

BufferPtr VectorFuzzer::fuzzIndices(
    vector_size_t size,
    vector_size_t baseVectorSize) {
  VELOX_CHECK_GE(size, 0);
  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool_);
  auto rawIndices = indices->asMutable<vector_size_t>();

  for (size_t i = 0; i < size; ++i) {
    rawIndices[i] = rand<vector_size_t>(rng_) % baseVectorSize;
  }
  return indices;
}

std::pair<int8_t, int8_t> VectorFuzzer::randPrecisionScale(
    int8_t maxPrecision) {
  // Generate precision in range [1, Decimal type max precision]
  auto precision = 1 + rand<int8_t>(rng_) % maxPrecision;
  // Generate scale in range [0, precision]
  auto scale = rand<int8_t>(rng_) % (precision + 1);
  return {precision, scale};
}

TypePtr VectorFuzzer::randType(int maxDepth) {
  return velox::randType(rng_, maxDepth);
}

TypePtr VectorFuzzer::randOrderableType(int maxDepth) {
  return velox::randOrderableType(rng_, maxDepth);
}

TypePtr VectorFuzzer::randOrderableType(
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return velox::randOrderableType(rng_, scalarTypes, maxDepth);
}

TypePtr VectorFuzzer::randType(
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return velox::randType(rng_, scalarTypes, maxDepth);
}

TypePtr VectorFuzzer::randMapType(int maxDepth) {
  return fuzzer::randMapType(rng_, defaultScalarTypes(), maxDepth);
}

RowTypePtr VectorFuzzer::randRowType(int maxDepth) {
  return velox::randRowType(rng_, maxDepth);
}

RowTypePtr VectorFuzzer::randRowType(
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return velox::randRowType(rng_, scalarTypes, maxDepth);
}

TypePtr VectorFuzzer::randRowTypeByWidth(
    const std::vector<TypePtr>& scalarTypes,
    int minWidth) {
  return velox::randRowTypeByWidth(rng_, scalarTypes, minWidth);
}

TypePtr VectorFuzzer::randRowTypeByWidth(int minWidth) {
  return velox::randRowTypeByWidth(rng_, defaultScalarTypes(), minWidth);
}

size_t VectorFuzzer::randInRange(size_t min, size_t max) {
  return rand(rng_, min, max);
}

VectorPtr VectorFuzzer::wrapInLazyVector(VectorPtr baseVector) {
  if (hasNestedDictionaryLayers(baseVector)) {
    auto indices = baseVector->wrapInfo();
    auto values = baseVector->valueVector();
    auto nulls = baseVector->nulls();

    auto copiedNulls = AlignedBuffer::copy(baseVector->pool(), nulls);

    return BaseVector::wrapInDictionary(
        copiedNulls, indices, baseVector->size(), wrapInLazyVector(values));
  }
  return std::make_shared<LazyVector>(
      baseVector->pool(),
      baseVector->type(),
      baseVector->size(),
      std::make_unique<VectorLoaderWrap>(baseVector));
}

RowVectorPtr VectorFuzzer::fuzzRowChildrenToLazy(RowVectorPtr rowVector) {
  std::vector<VectorPtr> children;
  VELOX_CHECK_NULL(rowVector->nulls());
  for (auto child : rowVector->children()) {
    VELOX_CHECK_NOT_NULL(child);
    VELOX_CHECK(!child->isLazy());
    // TODO: If child has dictionary wrappings, add ability to insert lazy wrap
    // between those layers.
    children.push_back(coinToss(0.5) ? wrapInLazyVector(child) : child);
  }
  return std::make_shared<RowVector>(
      pool_,
      rowVector->type(),
      nullptr,
      rowVector->size(),
      std::move(children));
}

// Utility function to check if a RowVector can have nested lazy children.
// This is only possible if the rows for its children map 1-1 with the top level
// rows (Rows of 'baseRowVector''s parent). For this to be true, the row vector
// cannot have any encoding layers over it, and it cannot have nulls.
bool canWrapRowChildInLazy(const VectorPtr& baseRowVector) {
  if (baseRowVector->typeKind() == TypeKind::ROW) {
    RowVector* rowVector = baseRowVector->as<RowVector>();
    if (rowVector) {
      return rowVector->nulls() == nullptr;
    }
  }
  return false;
}

// Utility function That only takes a row vector that passes the check in
// canWrapChildrenInLazy() and either picks the first child to be wrapped in
// lazy OR picks the first row vector child that passes canWrapChildrenInLazy()
// and traverses its children recursively.
VectorPtr wrapRowChildInLazyRecursive(VectorPtr& baseVector) {
  RowVector* rowVector = baseVector->as<RowVector>();
  VELOX_CHECK_NOT_NULL(rowVector);
  std::vector<VectorPtr> children;
  children.reserve(rowVector->childrenSize());
  bool foundChildVectorToWrap = false;
  for (column_index_t i = 0; i < rowVector->childrenSize(); i++) {
    auto child = rowVector->childAt(i);
    if (!foundChildVectorToWrap && canWrapRowChildInLazy(child)) {
      child = wrapRowChildInLazyRecursive(child);
      foundChildVectorToWrap = true;
    }
    children.push_back(child);
  }
  if (!foundChildVectorToWrap && !children.empty()) {
    children[0] = VectorFuzzer::wrapInLazyVector(children[0]);
  }

  BufferPtr newNulls = nullptr;
  if (rowVector->nulls()) {
    newNulls = AlignedBuffer::copy(rowVector->pool(), rowVector->nulls());
  }

  return std::make_shared<RowVector>(
      rowVector->pool(),
      rowVector->type(),
      std::move(newNulls),
      rowVector->size(),
      std::move(children));
}

RowVectorPtr VectorFuzzer::fuzzRowChildrenToLazy(
    RowVectorPtr rowVector,
    const std::vector<int>& columnsToWrapInLazy) {
  if (columnsToWrapInLazy.empty()) {
    return rowVector;
  }
  std::vector<VectorPtr> children;
  int listIndex = 0;
  for (column_index_t i = 0; i < rowVector->childrenSize(); i++) {
    auto child = rowVector->childAt(i);
    VELOX_USER_CHECK_NOT_NULL(child);
    VELOX_USER_CHECK(!child->isLazy());
    if (listIndex < columnsToWrapInLazy.size() &&
        i == (column_index_t)std::abs(columnsToWrapInLazy[listIndex])) {
      child = canWrapRowChildInLazy(child)
          ? wrapRowChildInLazyRecursive(child)
          : VectorFuzzer::wrapInLazyVector(child);
      if (columnsToWrapInLazy[listIndex] < 0) {
        // Negative index represents a lazy vector that is loaded.
        child->loadedVector();
      }
      listIndex++;
    }
    children.push_back(child);
  }

  BufferPtr newNulls = nullptr;
  if (rowVector->nulls()) {
    newNulls = AlignedBuffer::copy(rowVector->pool(), rowVector->nulls());
  }

  return std::make_shared<RowVector>(
      rowVector->pool(),
      rowVector->type(),
      std::move(newNulls),
      rowVector->size(),
      std::move(children));
}

VectorPtr VectorLoaderWrap::makeEncodingPreservedCopy(
    SelectivityVector& rows,
    vector_size_t vectorSize) {
  DecodedVector decoded;
  decoded.decode(*vector_, rows, false);

  if (decoded.isConstantMapping() || decoded.isIdentityMapping()) {
    VectorPtr result;
    BaseVector::ensureWritable(rows, vector_->type(), vector_->pool(), result);
    result->resize(vectorSize);
    result->copy(vector_.get(), rows, nullptr);
    return result;
  }

  SelectivityVector baseRows;
  auto baseVector = decoded.base();

  baseRows.resize(baseVector->size(), false);
  rows.applyToSelected([&](auto row) {
    if (!decoded.isNullAt(row)) {
      baseRows.setValid(decoded.index(row), true);
    }
  });
  baseRows.updateBounds();

  VectorPtr baseResult;
  BaseVector::ensureWritable(
      baseRows, baseVector->type(), vector_->pool(), baseResult);
  baseResult->copy(baseVector, baseRows, nullptr);

  BufferPtr indices = allocateIndices(vectorSize, vector_->pool());
  auto rawIndices = indices->asMutable<vector_size_t>();
  auto decodedIndices = decoded.indices();
  rows.applyToSelected(
      [&](auto row) { rawIndices[row] = decodedIndices[row]; });

  BufferPtr nulls = nullptr;
  if (decoded.nulls(&rows) || vectorSize > rows.end()) {
    // We fill [rows.end(), vectorSize) with nulls then copy nulls for selected
    // baseRows.
    nulls = allocateNulls(vectorSize, vector_->pool(), bits::kNull);
    if (baseRows.hasSelections()) {
      if (decoded.nulls(&rows)) {
        std::memcpy(
            nulls->asMutable<uint64_t>(),
            decoded.nulls(&rows),
            bits::nbytes(rows.end()));
      } else {
        bits::fillBits(
            nulls->asMutable<uint64_t>(), 0, rows.end(), bits::kNotNull);
      }
    }
  }

  return BaseVector::wrapInDictionary(
      std::move(nulls), std::move(indices), vectorSize, baseResult);
}

const std::vector<TypePtr>& defaultScalarTypes() {
  // @TODO Add decimal TypeKinds to randType.
  // Refer https://github.com/facebookincubator/velox/issues/3942
  static std::vector<TypePtr> kScalarTypes{
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
      TIMESTAMP(),
      DATE(),
      INTERVAL_DAY_TIME(),
  };
  return kScalarTypes;
}

TypePtr randType(FuzzerGenerator& rng, int maxDepth) {
  return randType(rng, defaultScalarTypes(), maxDepth);
}

TypePtr randType(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return fuzzer::randType(rng, scalarTypes, maxDepth);
}

TypePtr randMapType(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return fuzzer::randMapType(rng, scalarTypes, maxDepth);
}

TypePtr randTypeByWidth(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int minWidth) {
  if (minWidth <= 1) {
    const int numScalarTypes = scalarTypes.size();
    return scalarTypes[rand<uint32_t>(rng) % numScalarTypes];
  }

  switch (rand<uint32_t>(rng) % 3) {
    case 0:
      return ARRAY(randTypeByWidth(rng, scalarTypes, minWidth - 1));
    case 1: {
      const auto keyWidth =
          minWidth == 2 ? 1 : rand<uint32_t>(rng) % (minWidth - 2);
      return MAP(
          randTypeByWidth(rng, scalarTypes, keyWidth),
          randTypeByWidth(rng, scalarTypes, minWidth - keyWidth - 1));
    }
    // case 2:
    default:
      return randRowTypeByWidth(rng, scalarTypes, minWidth);
  }
}

TypePtr randRowTypeByWidth(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int minWidth) {
  const auto numFields = 1 + rand<uint32_t>(rng) % 10;
  std::vector<TypePtr> fields;
  auto remainingWidth = minWidth;
  for (auto i = 0; i < numFields - 1; ++i) {
    const auto fieldWidth =
        remainingWidth > 0 ? rand<uint32_t>(rng) % remainingWidth : 0;
    fields.push_back(randTypeByWidth(rng, scalarTypes, fieldWidth));
    remainingWidth -= fieldWidth;
  }
  fields.push_back(randTypeByWidth(rng, scalarTypes, remainingWidth));
  return ROW(std::move(fields));
}

TypePtr randOrderableType(FuzzerGenerator& rng, int maxDepth) {
  return randOrderableType(rng, defaultScalarTypes(), maxDepth);
}

TypePtr randOrderableType(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  // Should we generate a scalar type?
  if (maxDepth <= 1 || rand<bool>(rng)) {
    return randType(rng, scalarTypes, 0);
  }

  // ARRAY or ROW?
  if (rand<bool>(rng)) {
    return ARRAY(randOrderableType(rng, scalarTypes, maxDepth - 1));
  }

  auto numFields = 1 + rand<uint32_t>(rng) % 7;
  std::vector<std::string> names;
  std::vector<TypePtr> fields;
  for (int i = 0; i < numFields; ++i) {
    names.push_back(fmt::format("f{}", i));
    fields.push_back(randOrderableType(rng, scalarTypes, maxDepth - 1));
  }
  return ROW(std::move(names), std::move(fields));
}

RowTypePtr randRowType(FuzzerGenerator& rng, int maxDepth) {
  return randRowType(rng, defaultScalarTypes(), maxDepth);
}

RowTypePtr randRowType(
    FuzzerGenerator& rng,
    const std::vector<TypePtr>& scalarTypes,
    int maxDepth) {
  return fuzzer::randRowType(rng, scalarTypes, maxDepth);
}

} // namespace facebook::velox
