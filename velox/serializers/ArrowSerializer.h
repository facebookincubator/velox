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

#include <string_view>

#include "velox/vector/VectorStream.h"

namespace facebook::velox::serializer::arrow {

/// Serializes Velox vectors using Arrow IPC stream format. Uses the Arrow C
/// Data Interface bridge for zero-copy conversion where possible, then the
/// Arrow C++ IPC writer/reader for wire format serialization.
class ArrowVectorSerde : public VectorSerde {
 public:
  ArrowVectorSerde() : VectorSerde(std::string(kSerdeName)) {}

  std::unique_ptr<IterativeVectorSerializer> createIterativeSerializer(
      RowTypePtr type,
      int32_t numRows,
      StreamArena* streamArena,
      const Options* options) override;

  std::unique_ptr<BatchVectorSerializer> createBatchSerializer(
      memory::MemoryPool* pool,
      const Options* options) override;

  void deserialize(
      ByteInputStream* source,
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      const Options* options) override;

  /// Deserializes from an IOBuf directly with zero-copy semantics. Wraps the
  /// IOBuf memory as an Arrow buffer, avoiding the coalesce copy in the
  /// ByteInputStream path.
  void deserialize(
      const folly::IOBuf& source,
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      const Options* options) override;

  /// Registers as the global default vector serde.
  static void registerVectorSerde();

  /// Registers in the named serde registry under kSerdeName.
  static void registerNamedVectorSerde();

  /// Registers in the named serde registry only if not already registered.
  static void tryRegisterNamedVectorSerde();

  /// Registers a factory in the serde factory registry under kSerdeName.
  static void registerVectorSerdeFactory();

  static std::string_view name() {
    return kSerdeName;
  }

 private:
  // Identifier used for registry lookup.
  static constexpr std::string_view kSerdeName{"ArrowIpc"};
};

} // namespace facebook::velox::serializer::arrow
