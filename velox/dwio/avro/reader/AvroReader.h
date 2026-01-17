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

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace avro {
template <typename T>
class DataFileReader;
class GenericDatum;
} // namespace avro

namespace facebook::velox::avro {

struct AvroFileContents;

class AvroReader : public dwio::common::Reader {
 public:
  AvroReader(
      const std::unique_ptr<dwio::common::BufferedInput>& input,
      const dwio::common::ReaderOptions& options);

  std::optional<uint64_t> numberOfRows() const override;

  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t index) const override;

  const RowTypePtr& rowType() const override;

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override;

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

 private:
  std::shared_ptr<AvroFileContents> contents_;
};

} // namespace facebook::velox::avro
