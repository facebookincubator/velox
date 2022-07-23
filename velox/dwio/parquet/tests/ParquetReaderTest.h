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

//
// Created by Ying Su on 2/28/22.
//

#pragma once

//#include "velox/dwio/parquet/reader/ParquetReader.h"
//#include <gtest/gtest.h>
#include <dwio/common/Options.h>
#include <dwio/common/Reader.h>
#include <gtest/gtest.h>
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/VectorMaker.h"

using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox;

class ParquetReaderTest : public testing::Test {
 protected:
  std::string getExampleFilePath(const std::string& fileName) {
    return test::getDataFilePath(
        "velox/dwio/parquet/tests", "examples/" + fileName);
  }

  RowReaderOptions getReaderOpts(const RowTypePtr& rowType) {
    RowReaderOptions rowReaderOpts;
    rowReaderOpts.select(
        std::make_shared<ColumnSelector>(rowType, rowType->names()));

    return rowReaderOpts;
  }

  static RowTypePtr sampleSchema() {
    return ROW({"a", "b"}, {BIGINT(), DOUBLE()});
  }

  static RowTypePtr dateSchema() {
    return ROW({"date"}, {DATE()});
  }

  static RowTypePtr intSchema() {
    return ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  }

  std::shared_ptr<common::ScanSpec> makeScanSpec(const RowTypePtr& rowType) {
    auto scanSpec = std::make_shared<common::ScanSpec>("");

    for (auto i = 0; i < rowType->size(); ++i) {
      auto child =
          scanSpec->getOrCreateChild(common::Subfield(rowType->nameOf(i)));
      child->setProjectOut(true);
      child->setChannel(i);
    }

    return scanSpec;
  }

  std::unique_ptr<memory::ScopedMemoryPool> pool_{
      memory::getDefaultScopedMemoryPool()};
  std::unique_ptr<test::VectorMaker> vectorMaker_{
      std::make_unique<test::VectorMaker>(pool_.get())};
};
