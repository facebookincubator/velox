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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergConstantColumnFilter.h"

#include "velox/common/memory/Memory.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {
namespace {

class CudfIcebergConstantColumnFilterTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  ConstantFilterFold fold(
      const common::Filter& filter,
      const TypePtr& type,
      const std::optional<std::string>& value,
      bool readAsLocalTime = false) {
    return foldFilterOnConstant(
        filter, type, value, pool_.get(), readAsLocalTime);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(CudfIcebergConstantColumnFilterTest, varchar) {
  const common::BytesValues filter{{"apples"}, /*nullAllowed=*/false};

  EXPECT_EQ(fold(filter, VARCHAR(), "apples"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(filter, VARCHAR(), "oranges"), ConstantFilterFold::kAlwaysFalse);
}

TEST_F(CudfIcebergConstantColumnFilterTest, integers) {
  const common::BigintRange filter{5, 10, /*nullAllowed=*/false};

  EXPECT_EQ(fold(filter, BIGINT(), "5"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(fold(filter, INTEGER(), "10"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(fold(filter, SMALLINT(), "4"), ConstantFilterFold::kAlwaysFalse);
  EXPECT_EQ(fold(filter, TINYINT(), "11"), ConstantFilterFold::kAlwaysFalse);
}

TEST_F(CudfIcebergConstantColumnFilterTest, boolean) {
  const common::BoolValue filter{true, /*nullAllowed=*/false};

  EXPECT_EQ(fold(filter, BOOLEAN(), "true"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(fold(filter, BOOLEAN(), "false"), ConstantFilterFold::kAlwaysFalse);
}

TEST_F(CudfIcebergConstantColumnFilterTest, dateAsDaysSinceEpoch) {
  const auto days = DATE()->toDays("2025-06-05");
  const common::BigintRange filter{days, days, /*nullAllowed=*/false};

  // Iceberg-native encoding.
  EXPECT_EQ(
      fold(filter, DATE(), std::to_string(days)),
      ConstantFilterFold::kAlwaysTrue);
  // Hive-migrated encoding.
  EXPECT_EQ(
      fold(filter, DATE(), "2025-06-05"), ConstantFilterFold::kAlwaysTrue);
  EXPECT_EQ(
      fold(filter, DATE(), "2025-06-06"), ConstantFilterFold::kAlwaysFalse);
}

TEST_F(CudfIcebergConstantColumnFilterTest, nullValue) {
  const common::IsNull isNull;
  EXPECT_EQ(
      fold(isNull, BIGINT(), std::nullopt), ConstantFilterFold::kAlwaysTrue);

  const common::BigintRange rejectsNull{5, 10, /*nullAllowed=*/false};
  EXPECT_EQ(
      fold(rejectsNull, BIGINT(), std::nullopt),
      ConstantFilterFold::kAlwaysFalse);

  const common::BigintRange allowsNull{5, 10, /*nullAllowed=*/true};
  EXPECT_EQ(
      fold(allowsNull, BIGINT(), std::nullopt),
      ConstantFilterFold::kAlwaysTrue);
}

TEST_F(CudfIcebergConstantColumnFilterTest, unconvertibleValueIsUnknown) {
  const common::BigintRange filter{5, 10, /*nullAllowed=*/false};

  // Reported when the column is materialized, not here.
  EXPECT_EQ(fold(filter, BIGINT(), "apples"), ConstantFilterFold::kUnknown);
}

TEST_F(CudfIcebergConstantColumnFilterTest, unsupportedTypeIsUnknown) {
  const common::BigintRange filter{5, 10, /*nullAllowed=*/false};

  EXPECT_EQ(fold(filter, ARRAY(BIGINT()), "[5]"), ConstantFilterFold::kUnknown);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
