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

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/common/FormatData.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/dwrf/common/ByteRLE.h"
#include "velox/dwio/dwrf/common/Compression.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/reader/EncodingContext.h"
#include "velox/dwio/dwrf/reader/StripeStream.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::dwrf {

// DWRF specific functions shared between all readers.
class DwrfData : public FormatData {
 public:
  virtual std::vector<uint32_t> filterRowGroups(
      uint64_t rowsPerRowGroup,
      vonst StatsWriterInfo& context) {
    VELOX_NYI();
  }
};

};

// DWRF specific initialization.
class DwrfParams : public FormatParams {
 public:
  DwrfParams(
      memory::MemoryPool& pool,
      StripeStreams& stripe,
      FlatMapContext context = FlatMapContext::nonFlatMapContext())
      : FormatParams(pool), stripe_(stripe), flatMapContext_(context) {}

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& type) override {
    return nullptr;
  }

 private:
  StripeStreams& stripe_;
  FlatMapContext flatMapContext_;
};

} // namespace facebook::velox::dwrf
