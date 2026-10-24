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

// Copies 'decoded's row mapping for 'rows', leaving every other row on the zero
// allocateIndices() starts from. A dictionary leaves its indices unconstrained
// wherever it is null, and the vectors these indices go on to wrap hold no
// nulls that would keep an out of range one from being read.
BufferPtr copyIndices(
    const DecodedVector& decoded,
    const SelectivityVector& rows,
    memory::MemoryPool* pool) {
  auto indices = allocateIndices(decoded.size(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  rows.applyToSelected(
      [&](vector_size_t row) { rawIndices[row] = decoded.index(row); });
  return indices;
}

// Projects 'channels' out of a flat map's distinct keys. Scalar keys are
// copied, so that selecting a few channels out of a wide flat map does not
// retain the whole key vector. Complex keys are dictionary wrapped, where the
// copy costs more than the retention saves. Returns null when nothing was
// selected; FlatMapVector's constructor turns that into an empty keys vector.
VectorPtr selectDistinctKeys(
    const FlatMapVector& flatMap,
    const std::vector<column_index_t>& channels,
    memory::MemoryPool* pool) {
  if (channels.empty()) {
    return nullptr;
  }

  const auto& distinctKeys = flatMap.distinctKeys();
  const auto numSelected = static_cast<vector_size_t>(channels.size());

  if (!distinctKeys->type()->isPrimitiveType()) {
    auto indices = allocateIndices(numSelected, pool);
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t i = 0; i < numSelected; ++i) {
      rawIndices[i] = static_cast<vector_size_t>(channels[i]);
    }
    return BaseVector::wrapInDictionary(
        BufferPtr(nullptr), std::move(indices), numSelected, distinctKeys);
  }

  std::vector<BaseVector::CopyRange> ranges;
  ranges.reserve(numSelected);
  for (vector_size_t i = 0; i < numSelected; ++i) {
    ranges.push_back({static_cast<vector_size_t>(channels[i]), i, 1});
  }

  auto keys = BaseVector::create(distinctKeys->type(), numSelected, pool);
  keys->copyRanges(distinctKeys.get(), ranges);
  return keys;
}

// Handles a constant list of search keys over an unwrapped flat map. Every row
// selects the same channels, so the result keeps the input's map values and
// in-map buffers as they are, which makes the projection zero copy.
void applyFlatMap(
    const VectorPtr& keys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  // Row 0 describes the whole batch, but it has to be resolved through the
  // constant wrapping: indexing the base vector directly reads the wrong row.
  const auto& keysArray = *keys->wrappedVector()->asUnchecked<ArrayVector>();
  const auto keysIndex = keys->wrappedIndex(0);
  const auto begin = keysArray.offsetAt(keysIndex);
  const auto end = begin + keysArray.sizeAt(keysIndex);
  const auto& keyElements = keysArray.elements();

  std::vector<column_index_t> channels;
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;

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

    channels.push_back(*channel);
    mapValues.push_back(flatMap.mapValuesAt(*channel));
    if (*channel < flatMap.inMaps().size()) {
      inMaps.push_back(flatMap.inMaps()[*channel]);
    } else {
      // A missing in-map buffer means the key is present in every row, the
      // same convention FlatMapVector::isInMap() follows.
      inMaps.emplace_back(nullptr);
    }
  }

  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      flatMap.nulls(),
      flatMap.size(),
      selectDistinctKeys(flatMap, channels, context.pool()),
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
    indices = copyIndices(decodedMap, rows, context.pool());
  }

  std::vector<column_index_t> channels;
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;

  for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
    if (inMapsByChannel[channel] == nullptr) {
      continue;
    }
    channels.push_back(channel);
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

  // Top level nulls are left to the engine: map_subset is a default-null
  // function, so no row in 'rows' has a null map.
  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      nullptr,
      numRows,
      selectDistinctKeys(flatMap, channels, context.pool()),
      std::move(mapValues),
      std::move(inMaps));

  context.moveOrCopyResult(localResult, rows, result);
}

// Projects FlatMapVector key channels and delegates all other encodings to the
// existing simple-function adapter.
//
// Complex keys that contain a null compare differently between the two paths:
// the simple functions use kNullAsIndeterminate and throw, while
// FlatMapVector::getKeyChannel() compares nulls as values and matches. Flat map
// results follow the latter, which is the comparison a flat map already applies
// when deduplicating its distinct keys.
class MapSubsetVectorFunction : public exec::VectorFunction {
 public:
  explicit MapSubsetVectorFunction(
      std::shared_ptr<exec::VectorFunction> simpleFunction)
      : simpleFunction_{std::move(simpleFunction)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "map_subset expects a map and an array.");

    // Probing with wrappedVector() rather than a DecodedVector leaves the
    // MapVector path with the single decode the simple function already does.
    if (args[0]->wrappedVector()->encoding() !=
        VectorEncoding::Simple::FLAT_MAP) {
      simpleFunction_->apply(rows, args, outputType, context, result);
      return;
    }

    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    const auto& decodedMap = *mapDecoder.get();
    const auto& flatMap = *decodedMap.base()->asUnchecked<FlatMapVector>();

    if (args[1]->isConstantEncoding() && decodedMap.isIdentityMapping()) {
      applyFlatMap(args[1], flatMap, rows, outputType, context, result);
    } else {
      exec::LocalDecodedVector keysDecoder(context, *args[1], rows);
      applyFlatMap(
          decodedMap,
          *keysDecoder.get(),
          flatMap,
          rows,
          outputType,
          context,
          result);
    }
  }

 private:
  const std::shared_ptr<exec::VectorFunction> simpleFunction_;
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
  VELOX_CHECK(
      simpleFunction.has_value(),
      "Scalar function not registered for the given input types: {}",
      name);

  return std::make_shared<MapSubsetVectorFunction>(
      simpleFunction->createFunction()->createVectorFunction(
          inputTypes, constantInputs, config));
}

} // namespace

template <typename T>
void registerMapSubsetPrimitive(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapSubsetPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({name}, true, defaultOwner);
}

void registerMapSubset(const std::string& name, std::string_view defaultOwner) {
  registerMapSubsetPrimitive<bool>(name, defaultOwner);
  registerMapSubsetPrimitive<int8_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int16_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int32_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int64_t>(name, defaultOwner);
  registerMapSubsetPrimitive<float>(name, defaultOwner);
  registerMapSubsetPrimitive<double>(name, defaultOwner);
  registerMapSubsetPrimitive<Timestamp>(name, defaultOwner);
  registerMapSubsetPrimitive<Date>(name, defaultOwner);

  registerFunction<
      MapSubsetVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({name}, {}, true, defaultOwner);

  registerFunction<
      MapSubsetFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({name}, {}, true, defaultOwner);

  // The expression compiler prefers this vector function; its factory creates
  // the existing simple-function adapter as the non-flat-map fallback.
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
      },
      {},
      true,
      defaultOwner);
}
} // namespace facebook::velox::functions
