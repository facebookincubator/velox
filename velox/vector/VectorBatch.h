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
#pragma once

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/MappedMemory.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/StreamArena.h"
#include "velox/type/Type.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox {

class RowVector;

namespace batch {

enum class BatchSerdeStatus {
  Success,
  AllocationFailed,
  SerializationFailed,
  DeserializationFailed,
  KeysSerializationFailed,
  SizesDeserializationFailed
};

class VectorKeySerializer {
 public:
  /// Deserializes a set of keys in a vector an aggregated key
  /// \param vector the input row vector
  /// \param keys the indexes of the keys in the row vector
  /// \param row the row index
  /// \param serializedKeysBuffer a pre-allocated block for serialization
  /// \param serializedKeys the return serialized block
  /// \return the size of the serialized keys in the input block
  virtual std::optional<size_t> serializeKeys(
      const std::shared_ptr<RowVector>& vector,
      const std::vector<int>& keys,
      vector_size_t row,
      std::string_view& serializedKeysBuffer,
      std::string_view& serializedKeys) = 0;
  virtual ~VectorKeySerializer() = default;
};

class VectorSerde {
 public:
  VectorSerde(const std::unique_ptr<VectorKeySerializer>& keySerializer)
      : keySerializer_{std::move(keySerializer)} {}
  virtual ~VectorSerde() = default;

  /// Must be called every time a new vector will be processed
  /// \param vector the new vector to be serialized
  virtual void initialize(const std::shared_ptr<RowVector>& vector) = 0;

  /// Serialize a row from the input vector into the buffer
  /// \param vector input vector
  /// \param row the row index
  /// \param result the result buffer which will hold the serialized
  /// row (and key). This buffer is provided by the function and the next
  /// calls to the function can free or modify the buffer
  /// \return error code
  virtual BatchSerdeStatus serializeRow(
      const std::shared_ptr<RowVector>& vector,
      const vector_size_t row,
      std::string_view& serializedKeys,
      std::string_view& serializedRow) = 0;

  /// Block deserializer that converts a block of bytes to a vector given
  /// its row type
  /// \param values the row value pointers from which we create the vector
  /// \param type the vector type
  /// \param result the resulting vector
  /// \return
  virtual BatchSerdeStatus deserializeVector(
      const std::vector<std::optional<std::string_view>>& values,
      const std::shared_ptr<const RowType> type,
      std::shared_ptr<RowVector>* result) = 0;

 protected:
  const std::unique_ptr<VectorKeySerializer>& keySerializer_;
};

class VectorSerdeFactory {
 public:
  virtual std::unique_ptr<VectorSerde> createVectorSerde() = 0;

  virtual ~VectorSerdeFactory() = default;
};
} // namespace batch
} // namespace facebook::velox
