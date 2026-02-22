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

// Adapted from Apache Arrow.

#pragma once

#include <stdio.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "velox/dwio/parquet/writer/arrow/Exception.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"

namespace facebook::velox::parquet::arrow {

static constexpr int64_t DEFAULT_SCANNER_BATCH_SIZE = 128;

class PARQUET_EXPORT Scanner {
 public:
  explicit Scanner(
      std::shared_ptr<ColumnReader> reader,
      int64_t batchSize = DEFAULT_SCANNER_BATCH_SIZE,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : batchSize_(batchSize),
        levelOffset_(0),
        levelsBuffered_(0),
        valueBuffer_(allocateBuffer(pool)),
        valueOffset_(0),
        valuesBuffered_(0),
        reader_(std::move(reader)) {
    defLevels_.resize(descr()->maxDefinitionLevel() > 0 ? batchSize_ : 0);
    repLevels_.resize(descr()->maxRepetitionLevel() > 0 ? batchSize_ : 0);
  }

  virtual ~Scanner() {}

  static std::shared_ptr<Scanner> make(
      std::shared_ptr<ColumnReader> colReader,
      int64_t batchSize = DEFAULT_SCANNER_BATCH_SIZE,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  virtual void
  printNext(std::ostream& out, int width, bool withLevels = false) = 0;

  bool hasNext() {
    return levelOffset_ < levelsBuffered_ || reader_->hasNext();
  }

  const ColumnDescriptor* descr() const {
    return reader_->descr();
  }

  int64_t batchSize() const {
    return batchSize_;
  }

  void setBatchSize(int64_t batchSize) {
    batchSize_ = batchSize;
  }

 protected:
  int64_t batchSize_;

  std::vector<int16_t> defLevels_;
  std::vector<int16_t> repLevels_;
  int levelOffset_;
  int levelsBuffered_;

  std::shared_ptr<ResizableBuffer> valueBuffer_;
  int valueOffset_;
  int64_t valuesBuffered_;
  std::shared_ptr<ColumnReader> reader_;
};

template <typename DType>
class PARQUET_TEMPLATE_CLASS_EXPORT TypedScanner : public Scanner {
 public:
  typedef typename DType::CType T;

  explicit TypedScanner(
      std::shared_ptr<ColumnReader> reader,
      int64_t batchSize = DEFAULT_SCANNER_BATCH_SIZE,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : Scanner(std::move(reader), batchSize, pool) {
    typedReader_ = static_cast<TypedColumnReader<DType>*>(reader_.get());
    int valueByteSize = TypeTraits<DType::typeNum>::valueByteSize;
    PARQUET_THROW_NOT_OK(valueBuffer_->Resize(batchSize_ * valueByteSize));
    values_ = reinterpret_cast<T*>(valueBuffer_->mutable_data());
  }

  virtual ~TypedScanner() {}

  bool nextLevels(int16_t* defLevel, int16_t* repLevel) {
    if (levelOffset_ == levelsBuffered_) {
      levelsBuffered_ = static_cast<int>(typedReader_->readBatch(
          static_cast<int>(batchSize_),
          defLevels_.data(),
          repLevels_.data(),
          values_,
          &valuesBuffered_));

      valueOffset_ = 0;
      levelOffset_ = 0;
      if (!levelsBuffered_) {
        return false;
      }
    }
    *defLevel =
        descr()->maxDefinitionLevel() > 0 ? defLevels_[levelOffset_] : 0;
    *repLevel =
        descr()->maxRepetitionLevel() > 0 ? repLevels_[levelOffset_] : 0;
    levelOffset_++;
    return true;
  }

  bool next(T* val, int16_t* defLevel, int16_t* repLevel, bool* isNull) {
    if (levelOffset_ == levelsBuffered_) {
      if (!hasNext()) {
        // Out of data pages.
        return false;
      }
    }

    nextLevels(defLevel, repLevel);
    *isNull = *defLevel < descr()->maxDefinitionLevel();

    if (*isNull) {
      return true;
    }

    if (valueOffset_ == valuesBuffered_) {
      throw ParquetException("Value was non-null, but has not been buffered");
    }
    *val = values_[valueOffset_++];
    return true;
  }

  // Returns true if there is a next value.
  bool nextValue(T* val, bool* isNull) {
    if (levelOffset_ == levelsBuffered_) {
      if (!hasNext()) {
        // Out of data pages.
        return false;
      }
    }

    // Out of values.
    int16_t defLevel = -1;
    int16_t repLevel = -1;
    nextLevels(&defLevel, &repLevel);
    *isNull = defLevel < descr()->maxDefinitionLevel();

    if (*isNull) {
      return true;
    }

    if (valueOffset_ == valuesBuffered_) {
      throw ParquetException("Value was non-null, but has not been buffered");
    }
    *val = values_[valueOffset_++];
    return true;
  }

  virtual void
  printNext(std::ostream& out, int width, bool withLevels = false) {
    T val{};
    int16_t defLevel = -1;
    int16_t repLevel = -1;
    bool isNull = false;
    char buffer[80];

    if (!next(&val, &defLevel, &repLevel, &isNull)) {
      throw ParquetException("No more values buffered");
    }

    if (withLevels) {
      out << "  D:" << defLevel << " R:" << repLevel << " ";
      if (!isNull) {
        out << "V:";
      }
    }

    if (isNull) {
      std::string nullFmt = formatFwf<ByteArrayType>(width);
      snprintf(buffer, sizeof(buffer), nullFmt.c_str(), "NULL");
    } else {
      formatValue(&val, buffer, sizeof(buffer), width);
    }
    out << buffer;
  }

 private:
  // The ownership of this object is expressed through the reader_ variable in.
  // The base.
  TypedColumnReader<DType>* typedReader_;

  inline void formatValue(void* val, char* buffer, int bufsize, int width);

  T* values_;
};

template <typename DType>
inline void TypedScanner<DType>::formatValue(
    void* val,
    char* buffer,
    int bufsize,
    int width) {
  std::string fmt = formatFwf<DType>(width);
  snprintf(buffer, bufsize, fmt.c_str(), *reinterpret_cast<T*>(val));
}

template <>
inline void TypedScanner<Int96Type>::formatValue(
    void* val,
    char* buffer,
    int bufsize,
    int width) {
  std::string fmt = formatFwf<Int96Type>(width);
  std::string result = int96ToString(*reinterpret_cast<Int96*>(val));
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

template <>
inline void TypedScanner<ByteArrayType>::formatValue(
    void* val,
    char* buffer,
    int bufsize,
    int width) {
  std::string fmt = formatFwf<ByteArrayType>(width);
  std::string result = byteArrayToString(*reinterpret_cast<ByteArray*>(val));
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

template <>
inline void TypedScanner<FLBAType>::formatValue(
    void* val,
    char* buffer,
    int bufsize,
    int width) {
  std::string fmt = formatFwf<FLBAType>(width);
  std::string result = fixedLenByteArrayToString(
      *reinterpret_cast<FixedLenByteArray*>(val), descr()->typeLength());
  snprintf(buffer, bufsize, fmt.c_str(), result.c_str());
}

typedef TypedScanner<BooleanType> BoolScanner;
typedef TypedScanner<Int32Type> Int32Scanner;
typedef TypedScanner<Int64Type> Int64Scanner;
typedef TypedScanner<Int96Type> Int96Scanner;
typedef TypedScanner<FloatType> FloatScanner;
typedef TypedScanner<DoubleType> DoubleScanner;
typedef TypedScanner<ByteArrayType> ByteArrayScanner;
typedef TypedScanner<FLBAType> FixedLenByteArrayScanner;

template <typename RType>
int64_t scanAll(
    int32_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    uint8_t* values,
    int64_t* valuesBuffered,
    ColumnReader* reader) {
  typedef typename RType::T Type;
  auto typedReader = static_cast<RType*>(reader);
  auto vals = reinterpret_cast<Type*>(&values[0]);
  return typedReader->readBatch(
      batchSize, defLevels, repLevels, vals, valuesBuffered);
}

int64_t PARQUET_EXPORT scanAllValues(
    int32_t batchSize,
    int16_t* defLevels,
    int16_t* repLevels,
    uint8_t* values,
    int64_t* valuesBuffered,
    ColumnReader* reader);

} // namespace facebook::velox::parquet::arrow
