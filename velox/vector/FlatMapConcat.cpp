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
#include "velox/vector/FlatMapConcat.h"

#include <algorithm>

#include "velox/vector/DecodedVector.h"
#include "velox/vector/NullsBuilder.h"
#include "velox/vector/VectorMap.h"

namespace facebook::velox {
namespace {

void checkFlatMapInputsSupported(std::span<DecodedVector* const> inputs) {
  if (!std::ranges::all_of(inputs, [](const DecodedVector* input) {
        return input->isIdentityMapping();
      })) {
    VELOX_NYI(
        "flatMapConcat does not support encodings over FlatMapVector inputs.");
  }
}

struct OutputChannels {
  std::vector<VectorValueSetEntry> keys;
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;

  void reserve(vector_size_t numKeys) {
    keys.reserve(numKeys);
    mapValues.reserve(numKeys);
    inMaps.reserve(numKeys);
  }

  void push(VectorValueSetEntry key, VectorPtr values, BufferPtr inMap) {
    keys.push_back(key);
    mapValues.push_back(std::move(values));
    inMaps.push_back(std::move(inMap));
  }
};

vector_size_t totalNumDistinctKeys(std::span<DecodedVector* const> inputs) {
  vector_size_t numKeys = 0;
  for (auto* input : inputs) {
    numKeys += input->base()->asUnchecked<FlatMapVector>()->numDistinctKeys();
  }
  return numKeys;
}

BufferPtr concatNulls(
    memory::MemoryPool* pool,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  if (config.emptyForNull) {
    return nullptr;
  }

  NullsBuilder nullsBuilder(rows.end(), pool);
  for (auto* input : inputs) {
    nullsBuilder.addNulls(input->nulls(&rows));
  }
  return nullsBuilder.build();
}

BufferPtr buildRowMask(
    memory::MemoryPool* pool,
    const SelectivityVector& rows,
    const BufferPtr& nulls) {
  if (rows.isAllSelected() && !nulls) {
    return nullptr;
  }

  const auto numRows = rows.end();
  auto mask = allocateNulls(numRows, pool, bits::kNotNull);
  auto* rawMask = mask->asMutable<uint64_t>();
  if (!rows.isAllSelected()) {
    bits::andBits(rawMask, rows.asRange().bits(), 0, numRows);
  }
  if (nulls) {
    bits::andBits(rawMask, nulls->as<uint64_t>(), 0, numRows);
  }
  return mask;
}

// A null 'sourceInMap' means the key is in every row, so masking one has to
// start from all ones rather than skip.
BufferPtr buildInMap(
    memory::MemoryPool* pool,
    const BufferPtr& sourceInMap,
    const BufferPtr& mask,
    vector_size_t numRows) {
  if (!mask) {
    return sourceInMap;
  }

  auto inMap = AlignedBuffer::allocate<bool>(numRows, pool, false);
  auto* rawInMap = inMap->asMutable<uint64_t>();
  if (sourceInMap) {
    bits::copyBits(sourceInMap->as<uint64_t>(), 0, rawInMap, 0, numRows);
  } else {
    bits::fillBits(rawInMap, 0, numRows, true);
  }
  bits::andBits(rawInMap, mask->as<uint64_t>(), 0, numRows);
  return inMap;
}

OutputChannels collectOutputChannels(
    memory::MemoryPool* pool,
    std::span<DecodedVector* const> inputs,
    const BufferPtr& mask,
    vector_size_t numRows) {
  const auto numKeys = totalNumDistinctKeys(inputs);

  OutputChannels channels;
  channels.reserve(numKeys);

  VectorValueSet seen;
  seen.reserve(numKeys);

  for (auto* input : inputs) {
    const auto* flatMap = input->base()->asUnchecked<FlatMapVector>();
    const auto& keys = flatMap->distinctKeys();
    for (vector_size_t channel = 0; channel < flatMap->numDistinctKeys();
         ++channel) {
      const VectorValueSetEntry key{keys.get(), channel};
      if (!seen.insert(key).second) {
        VELOX_NYI(
            "Currently flatMapConcat does not support duplicate keys across "
            "inputs: {}",
            keys->toString(channel));
      }
      channels.push(
          key,
          flatMap->mapValuesAt(channel),
          buildInMap(pool, flatMap->inMapsAt(channel), mask, numRows));
    }
  }
  return channels;
}

VectorPtr buildDistinctKeys(
    memory::MemoryPool* pool,
    const TypePtr& keyType,
    const std::vector<VectorValueSetEntry>& keys) {
  const auto numKeys = static_cast<vector_size_t>(keys.size());
  auto distinctKeys = BaseVector::create(keyType, numKeys, pool);
  for (vector_size_t key = 0; key < numKeys; ++key) {
    distinctKeys->copy(keys[key].vector, key, keys[key].index, 1);
  }
  return distinctKeys;
}

} // namespace

bool allInputsAreFlatMap(std::span<DecodedVector* const> inputs) {
  const auto numFlatMap =
      std::ranges::count_if(inputs, [](const DecodedVector* input) {
        return input->base()->encoding() == VectorEncoding::Simple::FLAT_MAP;
      });

  if (numFlatMap == 0) {
    return false;
  }
  if (numFlatMap != static_cast<std::ptrdiff_t>(inputs.size())) {
    VELOX_NYI(
        "Concatenating a mix of MapVector and FlatMapVector inputs is not "
        "supported.");
  }
  return true;
}

FlatMapVectorPtr flatMapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config) {
  VELOX_CHECK_GT(inputs.size(), 0);
  checkFlatMapInputsSupported(inputs);

  const auto numRows = rows.end();
  auto newNulls = concatNulls(pool, inputs, rows, config);
  const auto mask = buildRowMask(pool, rows, newNulls);
  auto channels = collectOutputChannels(pool, inputs, mask, numRows);
  auto distinctKeys =
      buildDistinctKeys(pool, outputType->asMap().keyType(), channels.keys);

  return std::make_shared<FlatMapVector>(
      pool,
      outputType,
      std::move(newNulls),
      numRows,
      std::move(distinctKeys),
      std::move(channels.mapValues),
      std::move(channels.inMaps));
}

} // namespace facebook::velox
