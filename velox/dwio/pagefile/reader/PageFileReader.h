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


#include "velox/dwio/common/ReaderFactory.h"
#include "velox/vector/arrow/Bridge.h"
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>



namespace facebook::velox::pagefile {

// class ArrowRandomAccessFile : public ::arrow::io::RandomAccessFile {
// public:
//  ArrowRandomAccessFile(std::shared_ptr<dwio::common::BufferedInput> bufferedInput)
//      : bufferedInput_(std::move(bufferedInput)), bytesRead_(0) {
//   // Set up the buffer reader from BufferedInput
//   bufferReader_ = std::make_shared<arrow::io::BufferReader>(
//       bufferedInput_->getInputStream()->getReadFile()->data(),
//       bufferedInput_->getInputStream()->getReadFile()->size());
//  }
//
//  // Implement the required methods for RandomAccessFile
//  arrow::Result<int64_t> GetSize() override {
//   return bufferReader_->GetSize();
//  }
//
//  arrow::Status Close() override {
//   return bufferReader_->Close();
//  }
//
//  arrow::Result<int64_t> Tell() const override {
//   return bytesRead_;
//  }
//
//  bool closed() const override {
//   return bufferReader_->closed();
//  }
//
// arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
//   auto result = bufferReader_->Read(nbytes, out);
//   if (result.ok()) {
//    bytesRead_ += result.ValueOrDie();
//   }
//   return result;
//  }
//
//  arrow::Status Read(int64_t nbytes, std::shared_ptr<arrow::Buffer>* out) override {
//   auto result = bufferReader_->Read(nbytes, out);
//   if (result.ok()) {
//    bytesRead_ += result.ValueOrDie()->size();
//   }
//   return result.status();
//  }
//
//  arrow::Status Seek(int64_t position) override {
//   return bufferReader_->Seek(position);
//  }
//
// private:
//  std::shared_ptr<dwio::common::BufferedInput> bufferedInput_;
//  std::shared_ptr<arrow::io::BufferReader> bufferReader_;
//  int64_t bytesRead_;
// };

class ReaderBase {
 public:
  ReaderBase(
      std::unique_ptr<dwio::common::BufferedInput>,
      const dwio::common::ReaderOptions& options);

  virtual ~ReaderBase() = default;

  memory::MemoryPool& getMemoryPool() const {
    return pool_;
  }

  std::shared_ptr<velox::dwio::common::BufferedInput> bufferedInput() const {
    return input_;
  }

  uint64_t fileLength() const {
    return fileLength_;
  }

  const std::shared_ptr<const RowType>& schema() const {
    return schema_;
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& schemaWithId() {
    return schemaWithId_;
  }

  bool isFileColumnNamesReadAsLowerCase() const {
    return options_.fileColumnNamesReadAsLowerCase();
  }

  const tz::TimeZone* sessionTimezone() const {
    return options_.getSessionTimezone();
  }

  /// Returns the uncompressed size for columns in 'type' and its children in
  /// row group.
  int64_t rowGroupUncompressedSize(
      int32_t rowGroupIndex,
      const dwio::common::TypeWithId& type) const;

  /// Checks whether the specific row group has been loaded and
  /// the data still exists in the buffered inputs.
  bool isRowGroupBuffered(int32_t rowGroupIndex) const;

 private:
  // Reads and parses file footer.
  void loadFileMetaData();

  void initializeSchema();

  template <typename T>
  static std::shared_ptr<const RowType> createRowType(
      const std::vector<T>& children,
      bool fileColumnNamesReadAsLowerCase);

  memory::MemoryPool& pool_;
  const uint64_t footerEstimatedSize_;
  const uint64_t filePreloadThreshold_;
  // Copy of options. Must be owned by 'this'.
  const dwio::common::ReaderOptions options_;
  std::shared_ptr<velox::dwio::common::BufferedInput> input_;
  uint64_t fileLength_;
  RowTypePtr schema_;
  std::shared_ptr<const dwio::common::TypeWithId> schemaWithId_;

  const bool binaryAsString = false;

  // Map from row group index to pre-created loading BufferedInput.
  std::unordered_map<uint32_t, std::shared_ptr<dwio::common::BufferedInput>>
      inputs_;
};

class ArrowRandomAccessFile : public arrow::io::RandomAccessFile {
public:
    // Constructor to initialize with BufferedInput and Velox ReadFile
    ArrowRandomAccessFile(std::shared_ptr<velox::dwio::common::BufferedInput> bufferedInput)
        : bufferedInput_(bufferedInput), bytesRead_(0) {
        // Get the velox::ReadFile from the BufferedInput's stream
        veloxFile_ = bufferedInput_->getInputStream()->getReadFile();
    }

    // Implement GetSize method to return the size of the file
    arrow::Result<int64_t> GetSize() override {
        return veloxFile_->size();
    }

    // Implement Read method for reading nbytes
    arrow::Result<int64_t>  Read(int64_t nbytes, void* out) override {
        // Call pread to read the data into the provided buffer 'out'
        std::string_view result = veloxFile_->pread(currentPosition_, nbytes, out);

        // Check if the read was successful
        if (result.size() != nbytes) {
            return arrow::Status::IOError("Failed to read the requested number of bytes");
        }

        // Update the current position after the read
        currentPosition_ += result.size();
        bytesRead_ += result.size();

        // Return the number of bytes actually read
        return static_cast<int64_t>(result.size());
    }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
      // // Allocate a buffer to hold the data
      // ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateBuffer(nbytes));
      //
      // // Read data into the buffer using pread
      // std::string_view result = veloxFile_->pread(currentPosition_, nbytes, buffer->mutable_data());
      //
      // // Check if the read was successful
      // if (result.size() != nbytes) {
      //   return arrow::Status::IOError("Failed to read the requested number of bytes");
      // }
      //
      // // Update the current position after the read
      // currentPosition_ += result.size();
      // bytesRead_ += result.size();
      //
      // // Return the buffer
      // return buffer;
      return nullptr;
    }
    // Implement Seek method
    arrow::Status Seek(int64_t position) override {
        currentPosition_ = position;  // Track current seek position
        return arrow::Status::OK();
    }

    // Implement Tell method
    arrow::Result<int64_t> Tell() const override {
        return currentPosition_;
    }

    // Implement Close method
    arrow::Status Close() override {
        // Optionally close the underlying file if necessary
        return arrow::Status::OK();
    }

    // Return whether the file has been closed
    bool closed() const override {
        return false;  // You can track this if needed
    }

private:
    std::shared_ptr<velox::dwio::common::BufferedInput> bufferedInput_;  // Reference to BufferedInput
    std::shared_ptr<velox::ReadFile> veloxFile_;  // Velox file that we read from
    int64_t bytesRead_;  // Track bytes read
    int64_t currentPosition_;  // Track current file position
};


class PageFileRowReader :public dwio::common::RowReader {
 public:

 PageFileRowReader(
    const std::shared_ptr<ReaderBase> readerBase_,
   const dwio::common::RowReaderOptions& options);

 ~PageFileRowReader() override = default;

  uint64_t next(
      uint64_t size,
      velox::VectorPtr& result,
      const dwio::common::Mutation* mutation) override;
  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override{};
  void resetFilterCaches() override{};
  std::optional<size_t> estimatedRowSize() const override{};
  bool allPrefetchIssued() const override{};
  std::optional<std::vector<PrefetchUnit>> prefetchUnits() override{};
  int64_t nextRowNumber() override{};
  int64_t nextReadSize(uint64_t size) override{};

 void initializeIpcReader();
 void initializeReadRange();
 bool loadNextPage();


    private:
    const std::shared_ptr<ReaderBase> readerBase_;
    velox::memory::MemoryPool& pool_;
    const dwio::common::RowReaderOptions options_;
    std::shared_ptr<::arrow::io::RandomAccessFile> arrowRandomFile_;  // Source for reading Arrow IPC data
    std::shared_ptr<::arrow::ipc::RecordBatchFileReader> ipcReader_;
    std::shared_ptr<::arrow::RecordBatch> currentBatch_;  // Currently loaded batch
    int64_t currentPageIndex_;
    int64_t currentRowIndex_;

};

class PageFileReader : public dwio::common::Reader {
 public:
  /**
   * Constructor that lets the user specify reader options and input stream.
   */
  PageFileReader(
  std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options);

  static std::unique_ptr<Reader> create(
    std::unique_ptr<dwio::common::BufferedInput> input,
    const  dwio::common::ReaderOptions& options);

  ~PageFileReader() override = default;


  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t nodeId) const override {
    return nullptr;
  }
  //
  const std::shared_ptr<const RowType>& rowType() const override {
    return nullptr;
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override {
    return nullptr;
  }


  std::optional<uint64_t> numberOfRows() const override {
    // auto& fileFooter = readerBase_->getFooter();
    // if (fileFooter.hasNumberOfRows()) {
    //   return fileFooter.numberOfRows();
    // }
    return std::nullopt;
  }

  // static uint64_t getMemoryUse(
  //     ReaderBase& readerBase,
  //     int32_t stripeIx,
  //     const dwio::common::ColumnSelector& cs);
  //
  // uint64_t getMemoryUse(int32_t stripeIx = -1);
  //
  // uint64_t getMemoryUseByFieldId(
  //     const std::vector<uint64_t>& include,
  //     int32_t stripeIx = -1);
  //
  // uint64_t getMemoryUseByName(
  //     const std::vector<std::string>& names,
  //     int32_t stripeIx = -1);
  //
  // uint64_t getMemoryUseByTypeId(
  //     const std::vector<uint64_t>& include,
  //     int32_t stripeIx = -1);

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options =  {}) const override;

  // std::unique_ptr<DwrfRowReader> createDwrfRowReader(
  //     const dwio::common::RowReaderOptions& options = {}) const;

  /**
   * Create a reader to the for the dwrf file.
   * @param input the stream to read
   * @param options the options for reading the file
   */
  // static std::unique_ptr<DwrfReader> create(
  //     std::unique_ptr<dwio::common::BufferedInput> input,
  //     const dwio::common::ReaderOptions& options);
  //
  // ReaderBase* testingReaderBase() const {
  //   return readerBase_.get();
  // }

 private:
  // Ensures that files column names match the ones from the table schema using
  // column indices.
  void updateColumnNamesFromTableSchema();

 private:
  std::shared_ptr<ReaderBase> readerBase_;
};

class PageFileReaderFactory : public dwio::common::ReaderFactory {
 public:
  PageFileReaderFactory() : ReaderFactory(dwio::common::FileFormat::PAGEFILE) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options) override {
    return PageFileReader::create(std::move(input), options);
  }
};

} // namespace facebook::velox::dwrf
