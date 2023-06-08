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

#include <fmt/format.h>
#include <string>

#include "folly/Range.h"
#include "velox/common/caching/ScanTracker.h"
#include "velox/dwio/common/Common.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/StreamIdentifier.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/common/wrap/orc-proto-wrapper.h"

namespace facebook::velox::dwrf {

enum class DwrfFormat : uint8_t {
  kDwrf = 0,
  kOrc = 1,
};

// Writer version
constexpr folly::StringPiece WRITER_NAME_KEY{"orc.writer.name"};
constexpr folly::StringPiece WRITER_VERSION_KEY{"orc.writer.version"};
constexpr folly::StringPiece WRITER_HOSTNAME_KEY{"orc.writer.host"};
constexpr folly::StringPiece kDwioWriter{"dwio"};
constexpr folly::StringPiece kPrestoWriter{"presto"};

enum StripeCacheMode { NA = 0, INDEX = 1, FOOTER = 2, BOTH = 3 };

enum WriterVersion {
  ORIGINAL = 0, // all default versions including files written by Presto
  DWRF_4_9 = 1, // string stats collection uses text rather than string
  DWRF_5_0 = 2, // string stats saved/read as raw bytes,
  DWRF_6_0 = 3, // adding sequence to each stream
  DWRF_7_0 = 4, // flatmap integer dictionary sharing
  FUTURE = INT64_MAX
};

constexpr WriterVersion WriterVersion_CURRENT = WriterVersion::DWRF_7_0;

/**
 * Get the name of the WriterVersion.
 */
std::string writerVersionToString(WriterVersion kind);

// Stream kind of dwrf.
enum StreamKind {
  StreamKind_PRESENT = 0,
  StreamKind_DATA = 1,
  StreamKind_LENGTH = 2,
  StreamKind_DICTIONARY_DATA = 3,
  StreamKind_DICTIONARY_COUNT = 4,
  StreamKind_NANO_DATA = 5,
  StreamKind_ROW_INDEX = 6,
  StreamKind_IN_DICTIONARY = 7,
  StreamKind_STRIDE_DICTIONARY = 8,
  StreamKind_STRIDE_DICTIONARY_LENGTH = 9,
  StreamKind_BLOOM_FILTER_UTF8 = 10,
  StreamKind_IN_MAP = 11
};

// Stream kind of orc.
enum StreamKindOrc {
  StreamKindOrc_PRESENT = 0,
  StreamKindOrc_DATA = 1,
  StreamKindOrc_LENGTH = 2,
  StreamKindOrc_DICTIONARY_DATA = 3,
  StreamKindOrc_DICTIONARY_COUNT = 4,
  StreamKindOrc_SECONDARY = 5,
  StreamKindOrc_ROW_INDEX = 6,
  StreamKindOrc_BLOOM_FILTER = 7,
  StreamKindOrc_BLOOM_FILTER_UTF8 = 8,
  StreamKindOrc_ENCRYPTED_INDEX = 9,
  StreamKindOrc_ENCRYPTED_DATA = 10,
  StreamKindOrc_STRIPE_STATISTICS = 100,
  StreamKindOrc_FILE_STATISTICS = 101,

  StreamKindOrc_INVALID = -1
};

inline bool isIndexStream(StreamKind kind) {
  return kind == StreamKind::StreamKind_ROW_INDEX ||
      kind == StreamKind::StreamKind_BLOOM_FILTER_UTF8;
}

inline bool isIndexStream(StreamKindOrc kind) {
  return kind == StreamKindOrc::StreamKindOrc_ROW_INDEX ||
      kind == StreamKindOrc::StreamKindOrc_BLOOM_FILTER ||
      kind == StreamKindOrc::StreamKindOrc_BLOOM_FILTER_UTF8;
}

/**
 * Get the string representation of the StreamKind.
 */
// std::string streamKindToString(StreamKind kind);

class StreamInformation {
 public:
  virtual ~StreamInformation() = default;

  virtual StreamKind getKind() const = 0;
  virtual uint32_t getNode() const = 0;
  virtual uint32_t getSequence() const = 0;
  virtual uint64_t getOffset() const = 0;
  virtual uint64_t getLength() const = 0;
  virtual bool getUseVInts() const = 0;
  virtual bool valid() const = 0;

  // providing a default implementation otherwise leading to too much compiling
  // errors
  virtual StreamKindOrc getKindOrc() const {
    return StreamKindOrc_INVALID;
  }
};

enum ColumnEncodingKind {
  ColumnEncodingKind_DIRECT = 0,
  ColumnEncodingKind_DICTIONARY = 1,
  ColumnEncodingKind_DIRECT_V2 = 2,
  ColumnEncodingKind_DICTIONARY_V2 = 3
};

class DwrfStreamIdentifier;

class EncodingKey {
 public:
  static const EncodingKey& getInvalid() {
    static const EncodingKey INVALID;
    return INVALID;
  }

  uint32_t node;
  uint32_t sequence;

  EncodingKey()
      : EncodingKey(dwio::common::MAX_UINT32, dwio::common::MAX_UINT32) {}

  EncodingKey(uint32_t n, uint32_t s = 0) : node{n}, sequence{s} {}

  bool operator==(const EncodingKey& other) const {
    return node == other.node && sequence == other.sequence;
  }

  std::size_t hash() const {
    return std::hash<uint32_t>()(node) ^ std::hash<uint32_t>()(sequence);
  }

  bool valid() const {
    return node != dwio::common::MAX_UINT32 && sequence >= 0;
  }

  std::string toString() const {
    return fmt::format("[node={}, sequence={}]", node, sequence);
  }

  DwrfStreamIdentifier forKind(const proto::Stream_Kind kind) const;

  DwrfStreamIdentifier forKind(const proto::orc::Stream_Kind kind) const;
};

struct EncodingKeyHash {
  std::size_t operator()(const EncodingKey& ek) const {
    return ek.hash();
  }
};

class DwrfStreamIdentifier : public dwio::common::StreamIdentifier {
 public:
  static const DwrfStreamIdentifier& getInvalid() {
    static const DwrfStreamIdentifier INVALID;
    return INVALID;
  }

 public:
  DwrfStreamIdentifier()
      : column_(dwio::common::MAX_UINT32),
        format_(DwrfFormat::kDwrf),
        kind_(StreamKind_DATA) {}

  DwrfStreamIdentifier(const proto::Stream& stream)
      : DwrfStreamIdentifier(
            stream.node(),
            stream.has_sequence() ? stream.sequence() : 0,
            stream.has_column() ? stream.column() : dwio::common::MAX_UINT32,
            stream.kind()) {}

  DwrfStreamIdentifier(const proto::orc::Stream& stream)
      : DwrfStreamIdentifier(
            stream.column(),
            0,
            dwio::common::MAX_UINT32,
            stream.kind()) {}

  DwrfStreamIdentifier(
      uint32_t node,
      uint32_t sequence,
      uint32_t column,
      StreamKind kind)
      : StreamIdentifier(
            velox::cache::TrackingId((node << kNodeShift) | kind).id()),
        column_{column},
        format_(DwrfFormat::kDwrf),
        kind_(kind),
        encodingKey_{node, sequence} {}

  DwrfStreamIdentifier(
      uint32_t node,
      uint32_t sequence,
      uint32_t column,
      StreamKindOrc kind)
      : StreamIdentifier(
            velox::cache::TrackingId((node << kNodeShift) | kind).id()),
        column_{column},
        format_(DwrfFormat::kOrc),
        kindOrc_(kind),
        encodingKey_{node, sequence} {}

  DwrfStreamIdentifier(
      uint32_t node,
      uint32_t sequence,
      uint32_t column,
      proto::Stream_Kind pkind)
      : DwrfStreamIdentifier(
            node,
            sequence,
            column,
            static_cast<StreamKind>(pkind)) {}

  DwrfStreamIdentifier(
      uint32_t node,
      uint32_t sequence,
      uint32_t column,
      proto::orc::Stream_Kind pkind)
      : DwrfStreamIdentifier(
            node,
            sequence,
            column,
            static_cast<StreamKindOrc>(pkind)) {}

  ~DwrfStreamIdentifier() = default;

  bool operator==(const DwrfStreamIdentifier& other) const {
    // column == other.column may be join the expression if all files
    // share the same new version that has column field filled
    return encodingKey_ == other.encodingKey_ && kind_ == other.kind_;
  }

  std::size_t hash() const override {
    return encodingKey_.hash() ^ std::hash<uint32_t>()(kind_);
  }

  uint32_t column() const {
    return column_;
  }

  DwrfFormat format() const {
    return format_;
  }

  const StreamKind& kind() const {
    return kind_;
  }

  const StreamKindOrc& kindOrc() const {
    return kindOrc_;
  }

  const EncodingKey& encodingKey() const {
    return encodingKey_;
  }

  std::string toString() const override {
    return fmt::format(
        "[id={}, node={}, sequence={}, column={}, format={}, kind={}]",
        id_,
        encodingKey_.node,
        encodingKey_.sequence,
        column_,
        (uint32_t)format_,
        static_cast<uint32_t>(kind_));
  }

 private:
  static constexpr int32_t kNodeShift = 5;

  uint32_t column_;

  DwrfFormat format_;
  union {
    StreamKind kind_; // format_ == kDwrf
    StreamKindOrc kindOrc_; // format_ == kOrc
  };

  EncodingKey encodingKey_;
};

std::string columnEncodingKindToString(ColumnEncodingKind kind);

class StripeInformation {
 public:
  virtual ~StripeInformation() = default;

  /**
   * Get the byte offset of the start of the stripe.
   * @return the bytes from the start of the file
   */
  virtual uint64_t getOffset() const = 0;

  /**
   * Get the total length of the stripe in bytes.
   * @return the number of bytes in the stripe
   */
  virtual uint64_t getLength() const = 0;

  /**
   * Get the length of the stripe's indexes.
   * @return the number of bytes in the index
   */
  virtual uint64_t getIndexLength() const = 0;

  /**
   * Get the length of the stripe's data.
   * @return the number of bytes in the stripe
   */
  virtual uint64_t getDataLength() const = 0;

  /**
   * Get the length of the stripe's tail section, which contains its index.
   * @return the number of bytes in the tail
   */
  virtual uint64_t getFooterLength() const = 0;

  /**
   * Get the number of rows in the stripe.
   * @return a count of the number of rows
   */
  virtual uint64_t getNumberOfRows() const = 0;
};

enum RleVersion { RleVersion_1, RleVersion_2 };

constexpr int32_t RLE_MINIMUM_REPEAT = 3;
constexpr int32_t RLE_MAXIMUM_REPEAT = 127 + RLE_MINIMUM_REPEAT;
constexpr int32_t RLE_MAX_LITERAL_SIZE = 128;

} // namespace facebook::velox::dwrf
