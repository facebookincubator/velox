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

#include "velox/dwio/dwrf/writer/WriterBase.h"
#include "velox/common/process/ProcessBase.h"
#include "velox/dwio/dwrf/utils/ProtoUtils.h"

namespace facebook::velox::dwrf {

void WriterBase::writeFooter(const Type& type) {
  auto pos = writerSink_->size();
  footer_->setHeaderLength(ORC_MAGIC_LEN);
  footer_->setContentLength(pos - ORC_MAGIC_LEN);
  writerSink_->setMode(WriterSink::Mode::None);

  // write cache when available
  auto cacheSize = writerSink_->getCacheSize();
  if (cacheSize > 0) {
    writerSink_->writeCache();
    for (auto& i : writerSink_->getCacheOffsets()) {
      footer_->addStripeCacheOffsets(i);
    }
    pos = writerSink_->size();
  }

  ProtoUtils::writeType(type, *footer_);
  DWIO_ENSURE_EQ(footer_->typesSize(), footer_->statisticsSize());
  auto writerVersion =
      static_cast<uint32_t>(context_->getConfig(Config::WRITER_VERSION));
  writeUserMetadata(writerVersion);
  footer_->setNumberOfRows(context_->fileRowCount());
  footer_->setRowIndexStride(context_->indexStride());

  if (context_->fileRawSize() > 0 || context_->fileRowCount() == 0) {
    // ColumnTransformWriter, when rewriting presto written file does not have
    // rawSize.
    footer_->setRawDataSize(context_->fileRawSize());
  }
  auto* checksum = writerSink_->getChecksum();
  footer_->setCheckSumAlgorithm(
      (checksum != nullptr) ? checksum->getType()
                            : proto::ChecksumAlgorithm::NULL_);
  writeProto(footer_->getDwrfPtr());
  const auto footerLength = writerSink_->size() - pos;

  // write postscript
  pos = writerSink_->size();
  auto dwrfPostScript = ArenaCreate<proto::PostScript>(arena_.get());
  std::unique_ptr<PostScriptWriteWrapper> ps =
      std::make_unique<PostScriptWriteWrapper>(dwrfPostScript);
  ps->setWriterVersion(writerVersion);
  ps->setFooterLength(footerLength);
  ps->setCompression(context_->compression());
  if (context_->compression() !=
      common::CompressionKind::CompressionKind_NONE) {
    ps->setCompressionBlockSize(context_->compressionBlockSize());
  }

  ps->setCacheMode(writerSink_->getCacheMode());
  ps->setCacheSize(cacheSize);
  writeProto(ps, common::CompressionKind::CompressionKind_NONE);
  auto psLength = writerSink_->size() - pos;
  DWIO_ENSURE_LE(psLength, 0xff, "PostScript is too large: ", psLength);
  auto psLen = static_cast<char>(psLength);
  writerSink_->addBuffer(
      context_->getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM), &psLen, 1);
}

void WriterBase::writeUserMetadata(uint32_t writerVersion) {
  // add writer version
  userMetadata_[std::string{kWriterNameKey}] = kDwioWriter;
  userMetadata_[std::string{kWriterVersionKey}] =
      folly::to<std::string>(writerVersion);
  userMetadata_[std::string{kWriterHostnameKey}] = process::getHostName();
  std::for_each(userMetadata_.begin(), userMetadata_.end(), [&](auto& pair) {
    auto item = footer_->addMetadata();
    item.setName(pair.first);
    item.setValue(pair.second);
  });
}

void WriterBase::initBuffers() {
  context_->initBuffer();
  writerSink_->init(
      context_->getMemoryPool(MemoryUsageCategory::OUTPUT_STREAM));
}
} // namespace facebook::velox::dwrf
