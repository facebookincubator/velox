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

#include "velox/dwio/text/reader/TextReader.h"
#include <string>

namespace facebook::velox::text {

const std::string TEXTFILE_CODEC = "org.apache.hadoop.io.compress.GzipCodec";
const std::string TEXTFILE_COMPRESSION_EXTENSION = ".gz";
const std::string TEXTFILE_COMPRESSION_EXTENSION_RAW = ".deflate";

/// TODO: Add implementation
FileContents::FileContents(
    memory::MemoryPool& pool,
    const std::shared_ptr<const RowType>& /* t */)
    : pool{pool} {
  return;
}

/// TODO: Add implementation
RowReaderImpl::RowReaderImpl(
    std::shared_ptr<FileContents> fileContents,
    const RowReaderOptions& opts,
    bool prestoTextReader)
    : columnSelector_{ColumnSelector::apply(
          opts.selector(),
          fileContents->schema)},
      prestoTextReader_{prestoTextReader} {}

/// TODO: Add implementation
void RowReaderImpl::updateSelected() {
  return;
}

/// TODO: Add implementation
const ColumnSelector& RowReaderImpl::getColumnSelector() const {
  return columnSelector_;
}

/// TODO: Add implementation
std::shared_ptr<const TypeWithId> RowReaderImpl::getSelectedType() const {
  return nullptr;
}

/**
 * Create a response buffer matching the selected schema.
 */
/// TODO: Add implementation
std::unique_ptr<BaseVector> RowReaderImpl::createRowBatch(
    uint64_t /*capacity*/) const {
  return nullptr;
}

/**
 * Read the next batch of rows from the file.
 * Returns true if there is more data to be read
 * beyond what has been returned by this call, which
 * might be none.
 */
/// TODO: Add implementation
bool RowReaderImpl::next(BaseVector& /*data*/) {
  return false;
}

/// TODO: Add implementation
uint64_t RowReaderImpl::getRowNumber() const {
  return 0;
}

/// TODO: Add implementation
uint64_t RowReaderImpl::seekToRow(uint64_t /*rowNumber*/) {
  return 0;
}

/// TODO: Add implementation
uint64_t RowReaderImpl::skipRows(uint64_t /*numberOfRowsToSkip*/) {
  return 0;
}

/// TODO: Add implementation
bool RowReaderImpl::isSelectedField(
    const std::shared_ptr<const velox::dwio::common::TypeWithId>& /*t*/) {
  return false;
}

/// TODO: Add implementation
std::string RowReaderImpl::getString(
    RowReaderImpl& /*th*/,
    bool& /*isNull*/,
    DelimType& /*delim*/) {
  return "";
}

/// TODO: Add implementation
template <typename T>
T RowReaderImpl::getInteger(
    RowReaderImpl& /*th*/,
    bool& /*isNull*/,
    DelimType& /*delim*/) {
  return T();
}

/// TODO: Add implementation
bool RowReaderImpl::getBoolean(
    RowReaderImpl& /*th*/,
    bool& /*isNull*/,
    DelimType& /*delim*/) {
  return false;
}

/// TODO: Add implementation
float RowReaderImpl::getFloat(
    RowReaderImpl& /*th*/,
    bool& /*isNull*/,
    DelimType& /*delim*/) {
  return 0.0;
}

/// TODO: Add implementation
double RowReaderImpl::getDouble(
    RowReaderImpl& /*th*/,
    bool& /*isNull*/,
    DelimType& /*delim*/) {
  return 0;
}

/// TODO: Add implementation
template <class T, class reqT>
void RowReaderImpl::putValue(
    std::function<
        T(RowReaderImpl& /*th*/, bool& /*isNull*/, DelimType& delim)> /*f*/,
    BaseVector* FOLLY_NULLABLE /*data*/,
    DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
void RowReaderImpl::readElement(
    const std::shared_ptr<const Type>& /*t*/,
    const std::shared_ptr<const Type>& /*reqT*/,
    BaseVector* FOLLY_NULLABLE /*data*/,
    DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
const std::shared_ptr<const RowType>& RowReaderImpl::getType() const {
  static const std::shared_ptr<const RowType> dummy = nullptr;
  return dummy;
}

/// TODO: Add implementation
const char* RowReaderImpl::getStreamNameData() const {
  return "";
}

/// TODO: Add implementation
void RowReaderImpl::read(
    void* /*buf*/,
    uint64_t /*length*/,
    uint64_t /*offset*/,
    velox::dwio::common::LogType /*logType*/) {
  return;
}

/// TODO: Add implementation
uint64_t RowReaderImpl::getLength() {
  return 0;
}

/// TODO: Add implementation
uint64_t RowReaderImpl::getStreamLength() {
  return 0;
}

/// TODO: Add implementation
void RowReaderImpl::setEOF() {
  return;
}

/// TODO: Add implementation
void RowReaderImpl::incrementDepth() {
  return;
}

/// TODO: Add implementation
void RowReaderImpl::decrementDepth(DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
void RowReaderImpl::setEOE(DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
void RowReaderImpl::resetEOE(DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
bool RowReaderImpl::isEOE(DelimType /*delim*/) {
  return false;
}

/// TODO: Add implementation
void RowReaderImpl::setEOR(DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
bool RowReaderImpl::isEOR(DelimType /*delim*/) {
  return false;
}

/// TODO: Add implementation
bool RowReaderImpl::isOuterEOR(DelimType /*delim*/) {
  return false;
}

/// TODO: Add implementation
bool RowReaderImpl::isEOEorEOR(DelimType /*delim*/) {
  return false;
}

/// TODO: Add implementation
void RowReaderImpl::setNone(DelimType& /*delim*/) {
  return;
}

/// TODO: Add implementation
bool RowReaderImpl::isNone(DelimType /*delim*/) {
  return false;
}

/// TODO: Add implementation
DelimType RowReaderImpl::getDelimType(uint8_t /*v*/) {
  return DelimType();
}

/// TODO: Add implementation
uint8_t RowReaderImpl::getByte(DelimType& /*delim*/) {
  return 0;
}

/// TODO: Add implementation
bool RowReaderImpl::getEOR(DelimType& /*delim*/, bool& /*isNull*/) {
  return false;
}

/// TODO: Add implementation
bool RowReaderImpl::skipLine() {
  return false;
}

void RowReaderImpl::resetLine() {
  return;
}

uint64_t RowReaderImpl::incrementNumElements(BaseVector& /*data*/) {
  return 0;
}

/// TODO: Add implementation
ReaderImpl::ReaderImpl(
    std::unique_ptr<dwio::common::ReadFileInputStream> /*stream*/,
    const ReaderOptions& opts)
    : options_{opts} {
  return;
}

/// TODO: Add implementation
std::unique_ptr<RowReader> ReaderImpl::createRowReader() const {
  return nullptr;
}

/// TODO: Add implementation
std::unique_ptr<RowReader> ReaderImpl::createRowReader(
    const RowReaderOptions& /*opts*/) const {
  return nullptr;
}

/// TODO: Add implementation
std::unique_ptr<RowReader> ReaderImpl::createRowReader(
    const RowReaderOptions& /*opts*/,
    bool /*prestoTextReader*/) const {
  return nullptr;
}

/// TODO: Add implementation
common::CompressionKind ReaderImpl::getCompression() const {
  return contents_->compression;
}

/// TODO: Add implementation
uint64_t ReaderImpl::getNumberOfRows() const {
  return 0;
}

/// TODO: Add implementation
uint64_t ReaderImpl::getFileLength() const {
  return 0;
}

/// TODO: Add implementation
const std::shared_ptr<const RowType>& ReaderImpl::getType() const {
  static const std::shared_ptr<const RowType> dummy = nullptr;
  return dummy;
}

/// TODO: Add implementation
uint64_t ReaderImpl::getMemoryUse() {
  return 0;
}

} // namespace facebook::velox::text
