/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#pragma once

#include <memory>
#include <span>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class DictionaryEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  DictionaryEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Dictionary);
    const char* pos = data.data() + this->dataOffset_;
    alphabet_ = detail::createTypedEncodingView<T>(
        this->readSizedBytes(pos, "Dictionary alphabet"), this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(alphabet_);
    indices_ = detail::createTypedEncodingView<uint32_t>(
        {pos, this->remainingBytes(pos)}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(indices_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return alphabet_->readAt(indices_->readAt(index));
  }

  // Reads the alphabet indices of the rows, then gathers their values with one
  // read of the alphabet rather than one virtual read per row.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    ScopedVector<uint32_t> alphabetIndices{
        length, this->pool_, this->options_.bufferPool};
    indices_->read(offset, length, alphabetIndices.data());
    alphabet_->readAt(
        std::span<const uint32_t>{alphabetIndices.data(), length}, output);
  }

  // Does the same as readPhysical() for rows at arbitrary 'indices': reads
  // their alphabet indices, then gathers the values with one alphabet read.
  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    ScopedVector<uint32_t> alphabetIndices{
        indices.size(), this->pool_, this->options_.bufferPool};
    indices_->readAt(indices, alphabetIndices.data());
    alphabet_->readAt(
        std::span<const uint32_t>{alphabetIndices.data(), indices.size()},
        output);
  }

  std::unique_ptr<TypedEncodingView<T>> alphabet_;
  std::unique_ptr<TypedEncodingView<uint32_t>> indices_;
};

/// Reads a SharedDictionary encoding, which holds the alphabet index of each
/// row but not the alphabet. The alphabet belongs to the dictionary bound to
/// the value stream, and the caller passes it in
/// Encoding::Options::sharedDictionaryAlphabet.
template <typename T>
class SharedDictionaryEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  SharedDictionaryEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        alphabet_{options.sharedDictionaryAlphabet},
        indices_{detail::createTypedEncodingView<uint32_t>(
            data.substr(this->dataOffset_),
            this->pool_,
            options)} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::SharedDictionary);
    NIMBLE_USER_CHECK_NOT_NULL(
        alphabet_, "Shared dictionary encoding requires an alphabet.");
    NIMBLE_USER_CHECK_EQ(
        alphabet_->dataType(),
        TypeTraits<T>::dataType,
        "Shared dictionary alphabet has unexpected type.");
    NIMBLE_CHECK_FILE_EQ(
        indices_->dataType(),
        DataType::Uint32,
        "Shared dictionary indices have unexpected type.");
    NIMBLE_CHECK_FILE_EQ(
        indices_->rowCount(),
        this->rowCount_,
        "Shared dictionary index count differs from row count.");
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return detail::castFromPhysicalType<T>(
        alphabet_->template physicalValueAt<T>(indices_->readAt(index)));
  }

  // Reads the alphabet indices of the rows, then gathers their values with one
  // read of the alphabet rather than one read per row.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    ScopedVector<uint32_t> alphabetIndices{
        length, this->pool_, this->options_.bufferPool};
    indices_->read(offset, length, alphabetIndices.data());
    alphabet_->template materialize<T>(
        std::span<const uint32_t>{alphabetIndices.data(), length}, output);
  }

  // Does the same as readPhysical() for rows at arbitrary 'indices'.
  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    ScopedVector<uint32_t> alphabetIndices{
        indices.size(), this->pool_, this->options_.bufferPool};
    indices_->readAt(indices, alphabetIndices.data());
    alphabet_->template materialize<T>(
        std::span<const uint32_t>{alphabetIndices.data(), indices.size()},
        output);
  }

  // Shared with every chunk bound to the same dictionary. Keeps the strings
  // that reads return alive for the lifetime of the view.
  const std::shared_ptr<const SharedDictionaryAlphabet> alphabet_;
  // Reads the alphabet index of each row.
  const std::unique_ptr<TypedEncodingView<uint32_t>> indices_;
};

} // namespace facebook::nimble
