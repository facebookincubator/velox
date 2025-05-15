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

#include "velox/dwio/common/Writer.h"
#include "velox/dwio/common/WriterFactory.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::pagefile {

class CtePageWriter : public dwio::common::Writer {
 public:
  CtePageWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const dwio::common::WriterOptions& options,
      std::shared_ptr<memory::MemoryPool> pool);

  ~CtePageWriter() override = default;

  void write(const VectorPtr& input) override;

  // Forces the writer to flush, does not close the writer.
  void flush() override;

  bool finish() override {
    return true;
  }

  void close() override;

  void abort() override;

 private:
  std::unique_ptr<dwio::common::FileSink> sink_;
  const std::shared_ptr<const dwio::common::TypeWithId> schema_;
  std::shared_ptr<memory::MemoryPool> pool_;
  const dwio::common::WriterOptions options_;
  std::unique_ptr<VectorStreamGroup> batch_;
  VectorSerde* serde_;
};

class CtePageWriterFactory : public dwio::common::WriterFactory {
 public:
  CtePageWriterFactory() : WriterFactory(dwio::common::FileFormat::PAGEFILE) {}

  std::unique_ptr<dwio::common::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      const std::shared_ptr<dwio::common::WriterOptions>& options) override;

  std::unique_ptr<dwio::common::WriterOptions> createWriterOptions() override {
    return std::make_unique<dwio::common::WriterOptions>();
  }
};
} // namespace facebook::velox::pagefile
