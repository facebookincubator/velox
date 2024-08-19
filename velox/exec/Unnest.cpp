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

#include "velox/exec/Unnest.h"
#include "velox/common/base/Nulls.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {
Unnest::Unnest(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::UnnestNode>& unnestNode)
    : Operator(
          driverCtx,
          unnestNode->outputType(),
          operatorId,
          unnestNode->id(),
          "Unnest"),
      withOrdinality_(unnestNode->withOrdinality()),
      maxOutputSize_(outputBatchRows()) {
  const auto& inputType = unnestNode->sources()[0]->outputType();
  const auto& unnestVariables = unnestNode->unnestVariables();
  for (const auto& variable : unnestVariables) {
    if (!variable->type()->isArray() && !variable->type()->isMap()) {
      VELOX_UNSUPPORTED("Unnest operator supports only ARRAY and MAP types");
    }
    unnestChannels_.push_back(inputType->getChildIdx(variable->name()));
  }

  unnestDecoded_.resize(unnestVariables.size());

  if (withOrdinality_) {
    VELOX_CHECK_EQ(
        outputType_->children().back(),
        BIGINT(),
        "Ordinality column should be BIGINT type.");
  }

  column_index_t outputChannel = 0;
  for (const auto& variable : unnestNode->replicateVariables()) {
    identityProjections_.emplace_back(
        inputType->getChildIdx(variable->name()), outputChannel++);
  }
}

void Unnest::addInput(RowVectorPtr input) {
  input_ = std::move(input);

  for (auto& child : input_->children()) {
    child->loadedVector();
  }

  const auto size = input_->size();

  // The max number of elements at each row across all unnested columns.
  maxSizes_ = allocateSizes(size, pool());
  rawMaxSizes_ = maxSizes_->asMutable<vector_size_t>();

  rawSizes_.resize(unnestChannels_.size());
  rawOffsets_.resize(unnestChannels_.size());
  rawIndices_.resize(unnestChannels_.size());

  for (auto channel = 0; channel < unnestChannels_.size(); ++channel) {
    const auto& unnestVector = input_->childAt(unnestChannels_[channel]);

    auto& currentDecoded = unnestDecoded_[channel];
    currentDecoded.decode(*unnestVector);

    rawIndices_[channel] = currentDecoded.indices();

    if (unnestVector->typeKind() == TypeKind::ARRAY) {
      const auto* unnestBaseArray = currentDecoded.base()->as<ArrayVector>();
      rawSizes_[channel] = unnestBaseArray->rawSizes();
      rawOffsets_[channel] = unnestBaseArray->rawOffsets();
    } else {
      VELOX_CHECK(unnestVector->typeKind() == TypeKind::MAP);
      const auto* unnestBaseMap = currentDecoded.base()->as<MapVector>();
      rawSizes_[channel] = unnestBaseMap->rawSizes();
      rawOffsets_[channel] = unnestBaseMap->rawOffsets();
    }

    // Count max number of elements per row.
    auto* currentSizes = rawSizes_[channel];
    auto* currentIndices = rawIndices_[channel];
    for (auto row = 0; row < size; ++row) {
      if (!currentDecoded.isNullAt(row)) {
        const auto unnestSize = currentSizes[currentIndices[row]];
        if (rawMaxSizes_[row] < unnestSize) {
          rawMaxSizes_[row] = unnestSize;
        }
      }
    }
  }
}

RowVectorPtr Unnest::getOutput() {
  if (!input_) {
    return nullptr;
  }

  const auto size = input_->size();

  // Limit the number of input rows to keep output batch size within
  // 'maxOutputSize' if possible. Not process each input row fully when single
  // row's output exceeds maxOutputSize. Single row's output maybe into
  // multiple batches.
  vector_size_t numInput = 0;
  vector_size_t numElements = 0;
  vector_size_t partialProcessRowStartSize = -1;
  vector_size_t firstRowEndSize = -1;
  // Process first row.
  if (nextInputRow_ < size) {
    auto firstRow = nextInputRow_;
    firstRowEndSize = rawMaxSizes_[firstRow];
    vector_size_t remainingSize = firstRowEndSize - firstRowStartSize_;
    if (numElements + remainingSize > maxOutputSize_) {
      // Single row's output is into multiple batches.
      // Read the size range from them, not use 0 to rawMaxSizes_[row].
      firstRowEndSize = firstRowStartSize_ + maxOutputSize_ - numElements;
      // Process maxOutputSize_ in this getOutput.
      numElements = maxOutputSize_;
      partialProcessRowStartSize = firstRowEndSize;
    } else {
      // Not need to split this row
      numElements += remainingSize;
    }
    ++numInput;
  }
  // Not split middle row.
  // If there is only 1 row, the end row will not take effect, its startSize is
  // always 0.
  vector_size_t endRowEndSize = -1;
  for (auto row = nextInputRow_ + 1; row < size; ++row) {
    if (numElements >= maxOutputSize_) {
      break;
    }
    vector_size_t remainingSize = rawMaxSizes_[row];
    if (numElements + remainingSize > maxOutputSize_) {
      // This is the end row.
      // Single row's output is into multiple batches.
      // read the size range from them, not use 0 to rawMaxSizes_[row].
      endRowEndSize = maxOutputSize_ - numElements;
      // Process maxOutputSize_ in this getOutput.
      numElements = maxOutputSize_;
      partialProcessRowStartSize = endRowEndSize;
      ++numInput;
      break;
    } else {
      // Not split this row.
      numElements += remainingSize;
      ++numInput;
    }
  }
  // The end row is not partial, set it to the maxSize.
  if (endRowEndSize == -1 && numInput > 1) {
    endRowEndSize = rawMaxSizes_[nextInputRow_ + numInput - 1];
  }

  if (numElements == 0) {
    // All arrays/maps are null or empty.
    input_ = nullptr;
    nextInputRow_ = 0;
    return nullptr;
  }

  auto output = generateOutput(
      nextInputRow_, numInput, numElements, firstRowEndSize, endRowEndSize);
  if (partialProcessRowStartSize != -1) {
    firstRowStartSize_ = partialProcessRowStartSize;
    nextInputRow_ += numInput - 1;
  } else {
    firstRowStartSize_ = 0;
    nextInputRow_ += numInput;
  }

  if (nextInputRow_ >= size) {
    input_ = nullptr;
    nextInputRow_ = 0;
  }

  return output;
}

void Unnest::generateRepeatedColumns(
    vector_size_t start,
    vector_size_t size,
    vector_size_t numElements,
    std::vector<VectorPtr>& outputs,
    vector_size_t firstRowEndSize,
    vector_size_t endRowEndSize) {
  // Create "indices" buffer to repeat rows as many times as there are elements
  // in the array (or map) in unnestDecoded.
  auto repeatedIndices = allocateIndices(numElements, pool());
  auto* rawRepeatedIndices = repeatedIndices->asMutable<vector_size_t>();
  vector_size_t index = 0;
  if (size > 0) {
    for (auto i = firstRowStartSize_; i < firstRowEndSize; i++) {
      rawRepeatedIndices[index++] = start;
    }
  }
  for (auto row = start + 1; row < start + size - 1; ++row) {
    for (auto i = 0; i < rawMaxSizes_[row]; i++) {
      rawRepeatedIndices[index++] = row;
    }
  }
  if (size > 1) {
    for (auto i = 0; i < endRowEndSize; i++) {
      rawRepeatedIndices[index++] = start + size - 1;
    }
  }

  // Wrap "replicated" columns in a dictionary using 'repeatedIndices'.
  for (const auto& projection : identityProjections_) {
    outputs.at(projection.outputChannel) = BaseVector::wrapInDictionary(
        nullptr /*nulls*/,
        repeatedIndices,
        numElements,
        input_->childAt(projection.inputChannel));
  }
}

const Unnest::UnnestChannelEncoding Unnest::generateEncodingForChannel(
    column_index_t channel,
    vector_size_t start,
    vector_size_t size,
    vector_size_t numElements,
    vector_size_t firstRowEndSize,
    vector_size_t endRowEndSize) {
  BufferPtr elementIndices = allocateIndices(numElements, pool());
  auto* rawElementIndices = elementIndices->asMutable<vector_size_t>();

  auto nulls = allocateNulls(numElements, pool());
  auto rawNulls = nulls->asMutable<uint64_t>();

  auto& currentDecoded = unnestDecoded_[channel];
  auto* currentSizes = rawSizes_[channel];
  auto* currentOffsets = rawOffsets_[channel];
  auto* currentIndices = rawIndices_[channel];

  // Make dictionary index for elements column since they may be out of order.
  vector_size_t index = 0;
  bool identityMapping = true;
  if (firstRowStartSize_ != 0) {
    identityMapping = false;
  }

  auto firstEndRowGenerator = [&](vector_size_t row,
                                  vector_size_t startSize,
                                  vector_size_t endSize) {
    if (!currentDecoded.isNullAt(row)) {
      const auto offset = currentOffsets[currentIndices[row]];
      const auto unnestSize = currentSizes[currentIndices[row]];
      if (index != offset || endSize != rawMaxSizes_[row] ||
          unnestSize < endSize) {
        identityMapping = false;
      }
      auto currentUnnestSize = std::min(endSize, unnestSize);
      for (auto i = startSize; i < currentUnnestSize; i++) {
        rawElementIndices[index++] = offset + i;
      }

      for (auto i = std::max(startSize, currentUnnestSize); i < endSize; ++i) {
        bits::setNull(rawNulls, index++, true);
      }
    } else if (endSize - startSize > 0) {
      identityMapping = false;

      for (auto i = startSize; i < endSize; ++i) {
        bits::setNull(rawNulls, index++, true);
      }
    }
  };

  if (size > 0) {
    firstEndRowGenerator(start, firstRowStartSize_, firstRowEndSize);
  }

  for (auto row = start + 1; row < start + size - 1; ++row) {
    const auto maxSize = rawMaxSizes_[row];

    if (!currentDecoded.isNullAt(row)) {
      const auto offset = currentOffsets[currentIndices[row]];
      const auto unnestSize = currentSizes[currentIndices[row]];

      if (index != offset || unnestSize < maxSize) {
        identityMapping = false;
      }

      for (auto i = 0; i < unnestSize; i++) {
        rawElementIndices[index++] = offset + i;
      }

      for (auto i = unnestSize; i < maxSize; ++i) {
        bits::setNull(rawNulls, index++, true);
      }
    } else if (maxSize > 0) {
      identityMapping = false;

      for (auto i = 0; i < maxSize; ++i) {
        bits::setNull(rawNulls, index++, true);
      }
    }
  }

  if (size > 1) {
    firstEndRowGenerator(start + size - 1, 0, endRowEndSize);
  }

  return {elementIndices, nulls, identityMapping};
}

VectorPtr Unnest::generateOrdinalityVector(
    vector_size_t start,
    vector_size_t size,
    vector_size_t numElements,
    vector_size_t firstRowEndSize,
    vector_size_t endRowEndSize) {
  auto ordinalityVector =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numElements, pool());

  // Set the ordinality at each result row to be the index of the element in
  // the original array (or map) plus one.
  auto* rawOrdinality = ordinalityVector->mutableRawValues();
  if (size > 0) {
    const auto maxSize = firstRowEndSize - firstRowStartSize_;
    std::iota(rawOrdinality, rawOrdinality + maxSize, firstRowStartSize_ + 1);
    rawOrdinality += maxSize;
  }
  for (auto row = start + 1; row < start + size - 1; ++row) {
    const auto maxSize = rawMaxSizes_[row];
    std::iota(rawOrdinality, rawOrdinality + maxSize, 1);
    rawOrdinality += maxSize;
  }
  if (size > 1) {
    std::iota(rawOrdinality, rawOrdinality + endRowEndSize, 1);
    rawOrdinality += endRowEndSize;
  }

  return ordinalityVector;
}

RowVectorPtr Unnest::generateOutput(
    vector_size_t start,
    vector_size_t size,
    vector_size_t numElements,
    vector_size_t firstRowEndSize,
    vector_size_t endRowEndSize) {
  std::vector<VectorPtr> outputs(outputType_->size());
  generateRepeatedColumns(
      start, size, numElements, outputs, firstRowEndSize, endRowEndSize);

  // Create unnest columns.
  vector_size_t outputsIndex = identityProjections_.size();
  for (auto channel = 0; channel < unnestChannels_.size(); ++channel) {
    const auto unnestChannelEncoding = generateEncodingForChannel(
        channel, start, size, numElements, firstRowEndSize, endRowEndSize);

    auto& currentDecoded = unnestDecoded_[channel];
    if (currentDecoded.base()->typeKind() == TypeKind::ARRAY) {
      // Construct unnest column using Array elements wrapped using above
      // created dictionary.
      const auto* unnestBaseArray = currentDecoded.base()->as<ArrayVector>();
      outputs[outputsIndex++] =
          unnestChannelEncoding.wrap(unnestBaseArray->elements(), numElements);
    } else {
      // Construct two unnest columns for Map keys and values vectors wrapped
      // using above created dictionary.
      const auto* unnestBaseMap = currentDecoded.base()->as<MapVector>();
      outputs[outputsIndex++] =
          unnestChannelEncoding.wrap(unnestBaseMap->mapKeys(), numElements);
      outputs[outputsIndex++] =
          unnestChannelEncoding.wrap(unnestBaseMap->mapValues(), numElements);
    }
  }

  if (withOrdinality_) {
    // Ordinality column is always at the end.
    outputs.back() = generateOrdinalityVector(
        start, size, numElements, firstRowEndSize, endRowEndSize);
  }

  return std::make_shared<RowVector>(
      pool(), outputType_, BufferPtr(nullptr), numElements, std::move(outputs));
}

VectorPtr Unnest::UnnestChannelEncoding::wrap(
    const VectorPtr& base,
    vector_size_t wrapSize) const {
  if (identityMapping) {
    return base;
  }

  const auto result =
      BaseVector::wrapInDictionary(nulls, indices, wrapSize, base);

  // Dictionary vectors whose size is much smaller than the size of the
  // 'alphabet' (base vector) create efficiency problems downstream. For
  // example, expression evaluation may peel the dictionary encoding and
  // evaluate the expression on the base vector. If a dictionary references
  // 1K rows of the base vector with row numbers 1'000'000...1'000'999, the
  // expression needs to allocate a result vector of size 1'001'000. This
  // causes large number of large allocations and wastes a lot of
  // resources.
  //
  // Make a flat copy of the necessary rows to avoid inefficient downstream
  // processing.
  //
  // TODO A better fix might be to change expression evaluation (and all other
  // operations) to handle dictionaries with large alphabets efficiently.
  return BaseVector::copy(*result);
}

bool Unnest::isFinished() {
  return noMoreInput_ && input_ == nullptr;
}
} // namespace facebook::velox::exec
