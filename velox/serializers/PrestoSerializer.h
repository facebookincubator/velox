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

#include "velox/common/base/Crc.h"
#include "velox/common/compression/Compression.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::serializer::presto {

/// There are two ways to serialize data using PrestoVectorSerde:
///
/// 1. In order to append multiple RowVectors into the same serialized payload,
/// one can first create an IterativeVectorSerializer using
/// createIterativeSerializer(), then append successive RowVectors using
/// IterativeVectorSerializer::append(). In this case, since different RowVector
/// might encode columns differently, data is always flattened in the serialized
/// payload.
///
/// Note that there are two flavors of append(), one that takes a range of rows,
/// and one that takes a list of row ids. The former is useful when serializing
/// large sections of the input vector (or the full vector); the latter is
/// efficient for a selective subset, e.g. when splitting a vector to a large
/// number of output shuffle destinations.
///
/// 2. To serialize a single RowVector, one can use the BatchVectorSerializer
/// returned by createBatchSerializer(). Since it serializes a single RowVector,
/// it tries to preserve the encodings of the input data.
class PrestoVectorSerde : public VectorSerde {
 public:
  // Input options that the serializer recognizes.
  struct PrestoOptions : VectorSerde::Options {
    PrestoOptions() = default;

    PrestoOptions(
        bool _useLosslessTimestamp,
        common::CompressionKind _compressionKind,
        bool _nullsFirst = false)
        : VectorSerde::Options(_compressionKind),
          useLosslessTimestamp(_useLosslessTimestamp),
          nullsFirst(_nullsFirst) {}

    /// Currently presto only supports millisecond precision and the serializer
    /// converts velox native timestamp to that resulting in loss of precision.
    /// This option allows it to serialize with nanosecond precision and is
    /// currently used for spilling. Is false by default.
    bool useLosslessTimestamp{false};

    /// Serializes nulls of structs before the columns. Used to allow
    /// single pass reading of in spilling.
    ///
    /// TODO: Make Presto also serialize nulls before columns of
    /// structs.
    bool nullsFirst{false};

    /// Minimum achieved compression if compression is enabled. Compressing less
    /// than this causes subsequent compression attempts to be skipped. The more
    /// times compression misses the target the less frequently it is tried.
    float minCompressionRatio{0.8};
  };

  /// Adds the serialized sizes of the rows of 'vector' in 'ranges[i]' to
  /// '*sizes[i]'.
  void estimateSerializedSize(
      const BaseVector* vector,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& scratch) override;

  void estimateSerializedSize(
      const BaseVector* vector,
      const folly::Range<const vector_size_t*> rows,
      vector_size_t** sizes,
      Scratch& scratch) override;

  std::unique_ptr<IterativeVectorSerializer> createIterativeSerializer(
      RowTypePtr type,
      int32_t numRows,
      StreamArena* streamArena,
      const Options* options) override;

  /// Note that in addition to the differences highlighted in the VectorSerde
  /// interface, BatchVectorSerializer returned by this function can maintain
  /// the encodings of the input vectors recursively.
  std::unique_ptr<BatchVectorSerializer> createBatchSerializer(
      memory::MemoryPool* pool,
      const Options* options) override;

  bool supportsAppendInDeserialize() const override {
    return true;
  }

  void deserialize(
      ByteInputStream* source,
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      const Options* options) override {
    return deserialize(source, pool, type, result, 0, options);
  }

  void deserialize(
      ByteInputStream* source,
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      vector_size_t resultOffset,
      const Options* options) override;

  /// This function is used to deserialize a single column that is serialized in
  /// PrestoPage format. It is important to note that the PrestoPage format used
  /// here does not include the Presto page header. Therefore, the 'source'
  /// should contain uncompressed, serialized binary data, beginning at the
  /// column header.
  void deserializeSingleColumn(
      ByteInputStream* source,
      velox::memory::MemoryPool* pool,
      TypePtr type,
      VectorPtr* result,
      const Options* options);

  enum class TokenType {
    HEADER,
    NUM_COLUMNS,
    COLUMN_ENCODING,
    NUM_ROWS,
    NULLS,
    BYTE_ARRAY,
    SHORT_ARRAY,
    INT_ARRAY,
    LONG_ARRAY,
    INT128_ARRAY,
    VARIABLE_WIDTH_DATA_SIZE,
    VARIABLE_WIDTH_DATA,
    DICTIONARY_INDICES,
    DICTIONARY_ID,
    HASH_TABLE_SIZE,
    HASH_TABLE,
    NUM_FIELDS,
    OFFSETS,
  };

  struct Token {
    TokenType tokenType;
    uint32_t length;
  };

  /**
   * This function lexes the PrestoPage encoded source into tokens so that
   * Zstrong can parse the PrestoPage without knowledge of the PrestoPage
   * format. The compressor, which needs to parse presto page, uses this
   * function to attach meaning to each token in the source. Then the decoder
   * can simply regnerate the tokens and concatenate, so it is independent of
   * the PrestoPage format and agnostic to any changes in the format.
   *
   * @returns Status::OK() if the @p source successfully parses as a PrestoPage,
   * and fills @p out with the tokens. Otherwise, returns an error status and
   * does not modify @p out.
   *
   * WARNING: This function does not support compression, encryption, nulls
   * first, or lossless timestamps and will throw an exception if these features
   * are enabled.
   *
   * NOTE: If this function returns success, the lex is guaranteed to be valid.
   * However, if the source was not PrestoPage, this function may still return
   * success, if the source is also interpretable as a PrestoPage. It attempts
   * to validate as much as possible, to reduce false positives, but provides no
   * guarantees.
   */
  static Status lex(
      std::string_view source,
      std::vector<Token>& out,
      const Options* options = nullptr);

  static void registerVectorSerde();
};

class PrestoOutputStreamListener : public OutputStreamListener {
 public:
  void onWrite(const char* s, std::streamsize count) override {
    if (not paused_) {
      crc_.process_bytes(s, count);
    }
  }

  void pause() {
    paused_ = true;
  }

  void resume() {
    paused_ = false;
  }

  auto crc() const {
    return crc_;
  }

  void reset() {
    crc_.reset();
  }

 private:
  bool paused_{false};
  bits::Crc32 crc_;
};
} // namespace facebook::velox::serializer::presto
