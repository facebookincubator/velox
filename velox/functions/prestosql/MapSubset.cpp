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

#include "velox/functions/prestosql/MapSubset.h"
#include <folly/container/F14Set.h>
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::functions {
namespace {

// Wraps the flat map's distinct keys in a dictionary that exposes only the
// first 'numKeys' channels listed in 'keyIndices'. Returns nullptr when no key
// was selected; FlatMapVector's constructor turns that into an empty keys
// vector.
VectorPtr selectDistinctKeys(
    const FlatMapVector& flatMap,
    BufferPtr keyIndices,
    vector_size_t numKeys) {
  if (numKeys == 0) {
    return nullptr;
  }
  keyIndices->setSize(numKeys * sizeof(vector_size_t));
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr),
      std::move(keyIndices),
      numKeys,
      flatMap.distinctKeys());
}

BufferPtr copyIndices(const DecodedVector& decoded, memory::MemoryPool* pool) {
  auto indices = allocateIndices(decoded.size(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < decoded.size(); ++i) {
    rawIndices[i] = decoded.index(i);
  }
  return indices;
}

// Handles a constant list of search keys over an unwrapped flat map. Every row
// selects the same channels, so the result keeps the input's map values and
// in-map buffers as they are, which makes the projection zero copy.
void applyFlatMap(
    const DecodedVector& decodedKeys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();
  const auto& keyElements = keysArray.elements();

  // All rows search for the same keys, so the first selected row describes the
  // whole batch. The constant may point at any row of the base array, so the
  // range comes from that row's offset rather than from a fixed zero.
  vector_size_t begin{0};
  vector_size_t end{0};
  if (rows.hasSelections()) {
    const auto keysIndex = decodedKeys.index(rows.begin());
    begin = keysArray.offsetAt(keysIndex);
    end = begin + keysArray.sizeAt(keysIndex);
  }

  auto keyIndices = allocateIndices(end - begin, context.pool());
  auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;

  // Walking the search keys rather than the flat map's channels keeps this
  // proportional to the number of keys asked for, which is what makes the
  // projection cheap on maps with many distinct keys.
  folly::F14FastSet<column_index_t> selectedChannels;
  selectedChannels.reserve(end - begin);

  for (auto i = begin; i < end; ++i) {
    if (keyElements->isNullAt(i)) {
      continue;
    }
    const auto channel = flatMap.getKeyChannel(keyElements, i);
    // Duplicated search keys must not produce duplicated map entries. The
    // simple function implementations dedupe the same way, by collecting the
    // search keys into a set and walking the map once.
    if (!channel.has_value() || !selectedChannels.insert(*channel).second) {
      continue;
    }

    rawKeyIndices[mapValues.size()] = static_cast<vector_size_t>(*channel);
    mapValues.push_back(flatMap.mapValuesAt(*channel));
    if (*channel < flatMap.inMaps().size()) {
      inMaps.push_back(flatMap.inMaps()[*channel]);
    } else {
      // A missing in-map buffer means the key is present in every row, the
      // same convention FlatMapVector::isInMap() follows.
      inMaps.emplace_back(nullptr);
    }
  }

  // Evaluated before the vectors below are moved from.
  auto distinctKeys = selectDistinctKeys(
      flatMap,
      std::move(keyIndices),
      static_cast<vector_size_t>(mapValues.size()));
  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      flatMap.nulls(),
      flatMap.size(),
      std::move(distinctKeys),
      std::move(mapValues),
      std::move(inMaps));

  context.moveOrCopyResult(localResult, rows, result);
}

// Handles per-row search keys, a wrapped flat map, or both. Rows may ask for
// different keys and two of them may share a flat map row, so the in-map
// buffers are rebuilt in the output row space instead of being reused.
void applyFlatMap(
    const DecodedVector& decodedMap,
    const DecodedVector& decodedKeys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto numDistinctKeys = flatMap.numDistinctKeys();
  const auto numRows = decodedMap.size();
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();
  const auto& keyElements = keysArray.elements();

  std::vector<BufferPtr> inMapsByChannel(numDistinctKeys);

  rows.applyToSelected([&](vector_size_t row) {
    const auto mapRow = decodedMap.index(row);
    const auto keysIndex = decodedKeys.index(row);
    const auto begin = keysArray.offsetAt(keysIndex);
    const auto end = begin + keysArray.sizeAt(keysIndex);

    for (auto i = begin; i < end; ++i) {
      if (keyElements->isNullAt(i)) {
        continue;
      }
      const auto channel = flatMap.getKeyChannel(keyElements, i);
      if (!channel.has_value() || !flatMap.isInMap(*channel, mapRow)) {
        continue;
      }
      // Indexing by channel makes duplicated search keys idempotent.
      auto& inMap = inMapsByChannel[*channel];
      if (inMap == nullptr) {
        inMap = AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
      }
      bits::setBit(inMap->asMutable<uint64_t>(), row);
    }
  });

  BufferPtr indices;
  if (!decodedMap.isIdentityMapping()) {
    indices = copyIndices(decodedMap, context.pool());
  }

  auto keyIndices = allocateIndices(numDistinctKeys, context.pool());
  auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;

  for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
    if (inMapsByChannel[channel] == nullptr) {
      continue;
    }
    rawKeyIndices[mapValues.size()] = static_cast<vector_size_t>(channel);
    const auto& values = flatMap.mapValuesAt(channel);
    if (indices == nullptr) {
      mapValues.push_back(values);
    } else {
      mapValues.push_back(
          BaseVector::wrapInDictionary(
              BufferPtr(nullptr), indices, numRows, values));
    }
    inMaps.push_back(std::move(inMapsByChannel[channel]));
  }

  // Evaluated before the vectors below are moved from.
  auto distinctKeys = selectDistinctKeys(
      flatMap,
      std::move(keyIndices),
      static_cast<vector_size_t>(mapValues.size()));

  // Top level nulls are left to the engine: map_subset is a default-null
  // function, so no row in 'rows' has a null map.
  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      nullptr,
      numRows,
      std::move(distinctKeys),
      std::move(mapValues),
      std::move(inMaps));

  context.moveOrCopyResult(localResult, rows, result);
}

// Adds a FlatMapVector path to map_subset.
//
// On a flat map, map_subset is a projection of the map value channels whose
// keys were requested, so the result is built as a FlatMapVector that shares
// the input's map values instead of copying key/value pairs.
//
// The simple function implementations in MapSubset.h only understand
// MapVector - VectorReader<Map<K, V>> casts the decoded base to MapVector
// unchecked - so reaching them with a flat map crashes. This function shadows
// them in the expression compiler and delegates to them for every encoding
// other than FLAT_MAP.
class MapSubsetVectorFunction : public exec::VectorFunction {
 public:
  explicit MapSubsetVectorFunction(
      std::shared_ptr<exec::VectorFunction> mapVectorFunction)
      : mapVectorFunction_{std::move(mapVectorFunction)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "map_subset expects a map and an array.");

    // wrappedVector() peels dictionary, constant and sequence wrappings and
    // loads lazies through virtual dispatch, so the encoding is known without
    // materializing decoded indices. Delegating from here leaves the
    // MapVector path with the single decode the simple function does anyway.
    if (args[0]->wrappedVector()->encoding() !=
        VectorEncoding::Simple::FLAT_MAP) {
      mapVectorFunction_->apply(rows, args, outputType, context, result);
      return;
    }

    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    const auto& decodedMap = *mapDecoder.get();
    exec::LocalDecodedVector keysDecoder(context, *args[1], rows);
    const auto& decodedKeys = *keysDecoder.get();
    const auto& flatMap = *decodedMap.base()->asUnchecked<FlatMapVector>();

    if (decodedKeys.isConstantMapping() && decodedMap.isIdentityMapping()) {
      applyFlatMap(decodedKeys, flatMap, rows, outputType, context, result);
    } else {
      applyFlatMap(
          decodedMap, decodedKeys, flatMap, rows, outputType, context, result);
    }
  }

 private:
  const std::shared_ptr<exec::VectorFunction> mapVectorFunction_;
};

std::shared_ptr<exec::VectorFunction> makeMapSubset(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  std::vector<TypePtr> inputTypes;
  std::vector<VectorPtr> constantInputs;
  inputTypes.reserve(inputArgs.size());
  constantInputs.reserve(inputArgs.size());
  for (const auto& arg : inputArgs) {
    inputTypes.push_back(arg.type);
    constantInputs.push_back(arg.constantValue);
  }

  auto simpleFunction =
      exec::simpleFunctions().resolveFunction(name, inputTypes);
  VELOX_USER_CHECK(
      simpleFunction.has_value(),
      "Scalar function not registered for the given input types: {}",
      name);

  return std::make_shared<MapSubsetVectorFunction>(
      simpleFunction->createFunction()->createVectorFunction(
          inputTypes, constantInputs, config));
}

} // namespace

template <typename T>
void registerMapSubsetPrimitive(const std::string& name) {
  registerFunction<
      ParameterBinder<MapSubsetPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({name});
}

void registerMapSubset(const std::string& name) {
  registerMapSubsetPrimitive<bool>(name);
  registerMapSubsetPrimitive<int8_t>(name);
  registerMapSubsetPrimitive<int16_t>(name);
  registerMapSubsetPrimitive<int32_t>(name);
  registerMapSubsetPrimitive<int64_t>(name);
  registerMapSubsetPrimitive<float>(name);
  registerMapSubsetPrimitive<double>(name);
  registerMapSubsetPrimitive<Timestamp>(name);
  registerMapSubsetPrimitive<Date>(name);

  registerFunction<
      MapSubsetVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({name});

  registerFunction<
      MapSubsetFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({name});

  // Must come after the simple functions above: the vector function shadows
  // them in the expression compiler and resolves them as its fallback for
  // non-flat-map inputs.
  exec::registerStatefulVectorFunction(
      name,
      // map(K,V), array(K) -> map(K,V)
      {exec::FunctionSignatureBuilder()
           .typeVariable("K")
           .typeVariable("V")
           .returnType("map(K,V)")
           .argumentType("map(K,V)")
           .argumentType("array(K)")
           .build()},
      [name](
          const std::string& /*name*/,
          const std::vector<exec::VectorFunctionArg>& inputArgs,
          const core::QueryConfig& config) {
        return makeMapSubset(name, inputArgs, config);
      });
}
} // namespace facebook::velox::functions
