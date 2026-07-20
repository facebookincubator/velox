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

// Adapted from Apache Arrow.

// This module defines an abstract interface for iterating through pages in a
// Parquet column chunk within a row group. It could be extended in the future
// to iterate through all data pages in all chunks in a file.

#pragma once

#include <glog/logging.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "velox/dwio/parquet/writer/arrow/Statistics.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

// TODO: Parallel processing is not yet safe because of memory-ownership
// semantics (the PageReader may or may not own the memory referenced by a
// page).
//
// TODO(wesm): In the future Parquet implementations may store the crc code
// in facebook::velox::parquet::thrift::PageHeader. parquet-mr currently does
// not, so we also skip it here, both on the read and write path.
class Page {
 public:
  Page(const std::shared_ptr<::arrow::Buffer>& buffer, PageType::type type)
      : buffer_(buffer), type_(type) {}

  PageType::type type() const {
    return type_;
  }

  std::shared_ptr<::arrow::Buffer> buffer() const {
    return buffer_;
  }

  // @returns: A pointer to the page's data.
  const uint8_t* data() const {
    return buffer_->data();
  }

  // @returns: The total size in bytes of the page's data buffer.
  int32_t size() const {
    return static_cast<int32_t>(buffer_->size());
  }

 private:
  std::shared_ptr<::arrow::Buffer> buffer_;
  PageType::type type_;
};

/// \brief Base type for DataPageV1 and DataPageV2 including common attributes.
class DataPage : public Page {
 public:
  int32_t numValues() const {
    return numValues_;
  }
  Encoding::type encoding() const {
    return encoding_;
  }
  int64_t uncompressedSize() const {
    return uncompressedSize_;
  }
  const EncodedStatistics& statistics() const {
    return statistics_;
  }
  /// Return the row ordinal within the row group to the first row in the data
  /// page. Currently it is only present from data pages created by
  /// ColumnWriter in order to collect page index.
  std::optional<int64_t> firstRowIndex() const {
    return firstRowIndex_;
  }

  virtual ~DataPage() = default;

 protected:
  DataPage(
      PageType::type type,
      const std::shared_ptr<::arrow::Buffer>& buffer,
      int32_t numValues,
      Encoding::type encoding,
      int64_t uncompressedSize,
      const EncodedStatistics& statistics = EncodedStatistics(),
      std::optional<int64_t> firstRowIndex = std::nullopt)
      : Page(buffer, type),
        numValues_(numValues),
        encoding_(encoding),
        uncompressedSize_(uncompressedSize),
        statistics_(statistics),
        firstRowIndex_(std::move(firstRowIndex)) {}

  int32_t numValues_;
  Encoding::type encoding_;
  int64_t uncompressedSize_;
  EncodedStatistics statistics_;
  /// Row ordinal within the row group to the first row in the data page.
  std::optional<int64_t> firstRowIndex_;
};

class DataPageV1 : public DataPage {
 public:
  DataPageV1(
      const std::shared_ptr<::arrow::Buffer>& buffer,
      int32_t numValues,
      Encoding::type encoding,
      Encoding::type definitionLevelEncoding,
      Encoding::type repetitionLevelEncoding,
      int64_t uncompressedSize,
      const EncodedStatistics& statistics = EncodedStatistics(),
      std::optional<int64_t> firstRowIndex = std::nullopt)
      : DataPage(
            PageType::kDataPage,
            buffer,
            numValues,
            encoding,
            uncompressedSize,
            statistics,
            std::move(firstRowIndex)),
        definitionLevelEncoding_(definitionLevelEncoding),
        repetitionLevelEncoding_(repetitionLevelEncoding) {}

  Encoding::type repetitionLevelEncoding() const {
    return repetitionLevelEncoding_;
  }

  Encoding::type definitionLevelEncoding() const {
    return definitionLevelEncoding_;
  }

 private:
  Encoding::type definitionLevelEncoding_;
  Encoding::type repetitionLevelEncoding_;
};

class DataPageV2 : public DataPage {
 public:
  DataPageV2(
      const std::shared_ptr<::arrow::Buffer>& buffer,
      int32_t numValues,
      int32_t numNulls,
      int32_t numRows,
      Encoding::type encoding,
      int32_t definitionLevelsByteLength,
      int32_t repetitionLevelsByteLength,
      int64_t uncompressedSize,
      bool isCompressed = false,
      const EncodedStatistics& statistics = EncodedStatistics(),
      std::optional<int64_t> firstRowIndex = std::nullopt)
      : DataPage(
            PageType::kDataPageV2,
            buffer,
            numValues,
            encoding,
            uncompressedSize,
            statistics,
            std::move(firstRowIndex)),
        numNulls_(numNulls),
        numRows_(numRows),
        definitionLevelsByteLength_(definitionLevelsByteLength),
        repetitionLevelsByteLength_(repetitionLevelsByteLength),
        isCompressed_(isCompressed) {}

  int32_t numNulls() const {
    return numNulls_;
  }

  int32_t numRows() const {
    return numRows_;
  }

  int32_t definitionLevelsByteLength() const {
    return definitionLevelsByteLength_;
  }

  int32_t repetitionLevelsByteLength() const {
    return repetitionLevelsByteLength_;
  }

  bool isCompressed() const {
    return isCompressed_;
  }

 private:
  int32_t numNulls_;
  int32_t numRows_;
  int32_t definitionLevelsByteLength_;
  int32_t repetitionLevelsByteLength_;
  bool isCompressed_;
};

class DictionaryPage : public Page {
 public:
  DictionaryPage(
      const std::shared_ptr<::arrow::Buffer>& buffer,
      int32_t numValues,
      Encoding::type encoding,
      bool isSorted = false)
      : Page(buffer, PageType::kDictionaryPage),
        numValues_(numValues),
        encoding_(encoding),
        isSorted_(isSorted) {}

  int32_t numValues() const {
    return numValues_;
  }

  Encoding::type encoding() const {
    return encoding_;
  }

  bool isSorted() const {
    return isSorted_;
  }

 private:
  int32_t numValues_;
  Encoding::type encoding_;
  bool isSorted_;
};

} // namespace facebook::velox::parquet::arrow
