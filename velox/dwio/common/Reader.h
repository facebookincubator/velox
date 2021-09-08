/*
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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "velox/dwio/common/InputStream.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/dwrf/common/Statistics.h" // TODO: make dwio::common::ColumnStatistics
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::dwio::common {

// Abstract row reader interface. Used to parse a single data split
// into vectors.
class RowReader {
 public:
  virtual ~RowReader() = default;

  virtual uint64_t next(uint64_t size, velox::VectorPtr& result) = 0;

  virtual size_t estimatedRowSize() const = 0;
};

// Abstract reader interface used to parse data files.
// Provides basic file information and creates RowReader
// objects to access file data.
class Reader {
 public:
  virtual ~Reader() = default;

  // Get total number of rows in a file.
  virtual std::optional<uint64_t> getNumberOfRows() const = 0;

  virtual std::unique_ptr<velox::dwrf::ColumnStatistics> getColumnStatistics(
      uint32_t index) const = 0;

  virtual const velox::RowTypePtr& getType() const = 0;

  virtual const std::shared_ptr<const TypeWithId>& getTypeWithId() const = 0;

  virtual std::unique_ptr<RowReader> createRowReader(
      const RowReaderOptions& options = {}) const = 0;
};

} // namespace facebook::dwio::common
