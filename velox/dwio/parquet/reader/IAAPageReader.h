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

#include <folly/futures/Future.h>

#include "velox/dwio/common/compression/AsyncCompression.h"
#include "velox/dwio/parquet/reader/PageReader.h"

namespace facebook::velox::parquet {

using folly::SemiFuture;

struct PreDecompPageInfo {
  int64_t numValues; // Number of values in this row group
  int64_t visitedRows; // rows already read
  const char* FOLLY_NULLABLE dataPageData{nullptr};
  BufferPtr uncompressedData;
};

class IAAPageReader : public PageReader {
 public:
  IAAPageReader(
      std::unique_ptr<dwio::common::SeekableInputStream> stream,
      memory::MemoryPool& pool,
      ParquetTypeWithIdPtr fileType,
      common::CompressionKind codec,
      int64_t chunkSize)
      : PageReader(std::move(stream), pool, fileType, codec, chunkSize) {
    uncompressedDictData_ = nullptr;
    uncompressedDataV1Data_ = nullptr;
  }
  ~IAAPageReader();

  PageReaderType getType() {
    return PageReaderType::IAA;
  };

  /**
   * Submit decompression job to IAA, store the decompression future
   * @param need_pre_decompress true if the codec and window bits are adaptable
   * for IAA
   * @param numValues number of values in row group. This value is stored in
   * rowGroupPageInfo_ and used for determine whether there is need to
   * pre-decompress by IAA
   */
  void preDecompressPage(bool& need_pre_decompress, int64_t numValues);

  // Override method to call seekToPreDecompPage and
  // prefetchNextPage in IAAPageReader
  virtual void seekToPage(int64_t row);

 private:
  void prefetchDataPageV1(const thrift::PageHeader& pageHeader);
  void prefetchDataPageV2(const thrift::PageHeader& pageHeader);
  void prefetchDictionary(const thrift::PageHeader& pageHeader);

  // Prefetch the next page if there are more than one page in the row group
  // and then submit the page decompression job to IAA
  void prefetchNextPage();

  // Get decompressed page from IAA async decompressor. Then Reads and sets
  // 'rowOfPage_' and 'numRowsInPage_' and initializes a decoder for the found
  // page.
  bool seekToPreDecompPage(int64_t row);

  void prepareDataPageV1(
      const thrift::PageHeader& pageHeader,
      int64_t row,
      bool job_success = false);
  void prepareDictionary(
      const thrift::PageHeader& pageHeader,
      bool job_success = false);
  const bool iaaDecompress(
      const char* FOLLY_NONNULL pageData,
      uint32_t compressedSize,
      uint32_t uncompressedSize,
      BufferPtr& uncompressedData,
      SemiFuture<uint64_t>& future);

  // Used for pre-decompress
  BufferPtr uncompressedDictData_;
  BufferPtr uncompressedDataV1Data_;
  thrift::PageHeader dictPageHeader_;
  const char* FOLLY_NULLABLE dictPageData_{nullptr};
  bool needUncompressDict;

  thrift::PageHeader dataPageHeader_;
  const char* FOLLY_NULLABLE dataPageData_{nullptr};

  SemiFuture<uint64_t> dictDecompFuture = SemiFuture<uint64_t>::makeEmpty();
  SemiFuture<uint64_t> dataDecompFuture = SemiFuture<uint64_t>::makeEmpty();

  bool preDecompressDict_ = false;
  bool preDecompressData_ = false;
  bool isWinSizeFit_ = false;
  PreDecompPageInfo rowGroupPageInfo_;
};

} // namespace facebook::velox::parquet
