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
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::functions {
namespace {

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
    VELOX_CHECK_EQ(args.size(), 2);
    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    const auto& decodedMap = *mapDecoder.get();

    if (decodedMap.base()->encoding() != VectorEncoding::Simple::FLAT_MAP) {
      mapVectorFunction_->apply(rows, args, outputType, context, result);
      return;
    }

    exec::LocalDecodedVector keysDecoder(context, *args[1], rows);
    const auto& decodedKeys = *keysDecoder.get();
    const auto& flatMap = *decodedMap.base()->asUnchecked<FlatMapVector>();

    // An unwrapped flat map with a constant list of search keys selects the
    // same channels for every row, so the input's map values and in-map
    // buffers can be handed to the result as they are.
    if (decodedKeys.isConstantMapping() && decodedMap.isIdentityMapping()) {
      applyConstantKeys(
          decodedKeys, flatMap, rows, outputType, context, result);
    } else {
      applyPerRowKeys(
          decodedMap, decodedKeys, flatMap, rows, outputType, context, result);
    }
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // map(K,V), array(K) -> map(K,V)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("K")
                .typeVariable("V")
                .returnType("map(K,V)")
                .argumentType("map(K,V)")
                .argumentType("array(K)")
                .build()};
  }

 private:
  // Invokes 'onChannel' for every key of the array at 'arrayIndex' that the
  // flat map has a channel for. Null search keys and keys absent from the flat
  // map are skipped. The same channel may be reported more than once if the
  // search keys contain duplicates.
  template <typename TOnChannel>
  static void forEachRequestedChannel(
      const FlatMapVector& flatMap,
      const ArrayVector& keysArray,
      vector_size_t arrayIndex,
      TOnChannel onChannel) {
    const auto& keyElements = keysArray.elements();
    const auto begin = keysArray.offsetAt(arrayIndex);
    const auto end = begin + keysArray.sizeAt(arrayIndex);

    for (auto i = begin; i < end; ++i) {
      if (keyElements->isNullAt(i)) {
        continue;
      }
      if (const auto channel = flatMap.getKeyChannel(keyElements, i)) {
        onChannel(channel.value());
      }
    }
  }

  // Wraps the flat map's distinct keys in a dictionary that exposes only the
  // 'numKeys' first channels listed in 'keyIndices'. Returns nullptr when no
  // key was selected; FlatMapVector's constructor turns that into an empty
  // keys vector.
  static VectorPtr selectDistinctKeys(
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

  static BufferPtr copyIndices(
      const DecodedVector& decoded,
      memory::MemoryPool* pool) {
    auto indices = allocateIndices(decoded.size(), pool);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < decoded.size(); ++i) {
      rawIndices[i] = decoded.index(i);
    }
    return indices;
  }

  // Builds the result by keeping the requested channels of the flat map as
  // they are, which makes the projection zero copy.
  void applyConstantKeys(
      const DecodedVector& decodedKeys,
      const FlatMapVector& flatMap,
      const SelectivityVector& rows,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    const auto numDistinctKeys = flatMap.numDistinctKeys();
    std::vector<bool> requested(numDistinctKeys, false);

    if (rows.hasSelections()) {
      forEachRequestedChannel(
          flatMap,
          *decodedKeys.base()->asUnchecked<ArrayVector>(),
          decodedKeys.index(rows.begin()),
          [&](column_index_t channel) { requested[channel] = true; });
    }

    auto keyIndices = allocateIndices(numDistinctKeys, context.pool());
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
    std::vector<VectorPtr> mapValues;
    std::vector<BufferPtr> inMaps;
    vector_size_t numKeys = 0;

    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      if (!requested[channel]) {
        continue;
      }
      rawKeyIndices[numKeys++] = channel;
      mapValues.push_back(flatMap.mapValuesAt(channel));
      inMaps.push_back(
          channel < flatMap.inMaps().size() ? flatMap.inMaps()[channel]
                                            : nullptr);
    }

    auto localResult = std::make_shared<FlatMapVector>(
        context.pool(),
        outputType,
        flatMap.nulls(),
        flatMap.size(),
        selectDistinctKeys(flatMap, std::move(keyIndices), numKeys),
        std::move(mapValues),
        std::move(inMaps));

    context.moveOrCopyResult(localResult, rows, result);
  }

  // Builds the result directly in the output row space: rows may ask for
  // different keys, and two of them may share a flat map row, so the in-map
  // buffers cannot be indexed by the flat map's rows.
  void applyPerRowKeys(
      const DecodedVector& decodedMap,
      const DecodedVector& decodedKeys,
      const FlatMapVector& flatMap,
      const SelectivityVector& rows,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const {
    const auto numDistinctKeys = flatMap.numDistinctKeys();
    const auto numRows = decodedMap.size();
    const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();

    std::vector<BufferPtr> inMapsByChannel(numDistinctKeys);

    rows.applyToSelected([&](vector_size_t row) {
      const auto mapRow = decodedMap.index(row);
      forEachRequestedChannel(
          flatMap,
          keysArray,
          decodedKeys.index(row),
          [&](column_index_t channel) {
            if (!flatMap.isInMap(channel, mapRow)) {
              return;
            }
            auto& inMap = inMapsByChannel[channel];
            if (inMap == nullptr) {
              inMap =
                  AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
            }
            bits::setBit(inMap->asMutable<uint64_t>(), row);
          });
    });

    BufferPtr indices;
    if (!decodedMap.isIdentityMapping()) {
      indices = copyIndices(decodedMap, context.pool());
    }

    auto keyIndices = allocateIndices(numDistinctKeys, context.pool());
    auto* rawKeyIndices = keyIndices->asMutable<vector_size_t>();
    std::vector<VectorPtr> mapValues;
    std::vector<BufferPtr> inMaps;
    vector_size_t numKeys = 0;

    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      if (inMapsByChannel[channel] == nullptr) {
        continue;
      }
      rawKeyIndices[numKeys++] = channel;
      const auto& values = flatMap.mapValuesAt(channel);
      mapValues.push_back(
          indices == nullptr
              ? values
              : BaseVector::wrapInDictionary(
                    BufferPtr(nullptr), indices, numRows, values));
      inMaps.push_back(std::move(inMapsByChannel[channel]));
    }

    // Top level nulls are left to the engine: map_subset is a default-null
    // function, so no row in 'rows' has a null map.
    auto localResult = std::make_shared<FlatMapVector>(
        context.pool(),
        outputType,
        nullptr,
        numRows,
        selectDistinctKeys(flatMap, std::move(keyIndices), numKeys),
        std::move(mapValues),
        std::move(inMaps));

    context.moveOrCopyResult(localResult, rows, result);
  }

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
      MapSubsetVectorFunction::signatures(),
      [name](
          const std::string& /*name*/,
          const std::vector<exec::VectorFunctionArg>& inputArgs,
          const core::QueryConfig& config) {
        return makeMapSubset(name, inputArgs, config);
      });
}
} // namespace facebook::velox::functions
