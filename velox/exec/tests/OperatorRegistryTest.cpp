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

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/exec/Driver.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
namespace {

// Stand-in registry value; the operator-scoped registry stores arbitrary
// types keyed by subsystem name.
struct TierConfig {
  std::string name;
};

constexpr std::string_view kTestKey{"testKey"};

// Observational DriverAdapter: runs the test-provided hook against the driver
// during adaptation and does not replace any operators.
using DriverHook = std::function<void(exec::Driver&)>;
DriverHook* gDriverHook{nullptr};

bool testAdapter(const exec::DriverFactory& /*factory*/, exec::Driver& driver) {
  if (gDriverHook != nullptr && *gDriverHook) {
    (*gDriverHook)(driver);
  }
  return false;
}

class OperatorRegistryTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedAdapters_ = DriverFactory::adapters;
    DriverFactory::adapters.clear();
    DriverFactory::registerAdapter(
        exec::DriverAdapter{"operatorRegistryTest", {}, testAdapter});
  }

  void TearDown() override {
    DriverFactory::adapters = savedAdapters_;
    gDriverHook = nullptr;
    OperatorTestBase::TearDown();
  }

  // Runs a two-operator (Values -> FilterProject) single-driver plan with
  // 'hook' installed as the adapter body and 'ctx' as the query context.
  void run(DriverHook hook, const std::shared_ptr<core::QueryCtx>& ctx) {
    gDriverHook = &hook;
    auto data = makeRowVector(
        {makeFlatVector<int32_t>(10, [](auto row) { return row; })});
    auto plan = PlanBuilder().values({data}).filter("c0 >= 0").planNode();
    AssertQueryBuilder(plan).queryCtx(ctx).maxDrivers(1).copyResults(pool());
    gDriverHook = nullptr;
  }

  std::shared_ptr<core::QueryCtx> makeQueryCtx() {
    return core::QueryCtx::create(driverExecutor_.get());
  }

  std::vector<DriverAdapter> savedAdapters_;
};

TEST_F(OperatorRegistryTest, queryLevelEntryVisibleToOperator) {
  auto ctx = makeQueryCtx();
  auto queryValue = std::make_shared<TierConfig>(TierConfig{"query"});
  ctx->setRegistry<TierConfig>(kTestKey, queryValue);

  std::shared_ptr<TierConfig> seen;
  run(
      [&](exec::Driver& driver) {
        auto operators = driver.operators();
        ASSERT_GE(operators.size(), 1);
        seen = operators[0]->registry<TierConfig>(kTestKey);
      },
      ctx);

  EXPECT_EQ(seen, queryValue);
}

TEST_F(OperatorRegistryTest, operatorLevelOverrideShadowsQueryLevel) {
  auto ctx = makeQueryCtx();
  auto queryValue = std::make_shared<TierConfig>(TierConfig{"query"});
  ctx->setRegistry<TierConfig>(kTestKey, queryValue);
  auto operatorValue = std::make_shared<TierConfig>(TierConfig{"operator"});

  std::shared_ptr<TierConfig> overridden;
  std::shared_ptr<TierConfig> sibling;
  run(
      [&](exec::Driver& driver) {
        auto operators = driver.operators();
        ASSERT_GE(operators.size(), 2);
        operators[0]->setRegistry<TierConfig>(kTestKey, operatorValue);
        overridden = operators[0]->registry<TierConfig>(kTestKey);
        sibling = operators[1]->registry<TierConfig>(kTestKey);
      },
      ctx);

  EXPECT_EQ(overridden, operatorValue);
  EXPECT_EQ(sibling, queryValue);
}

TEST_F(OperatorRegistryTest, missingEntryReturnsNullptr) {
  auto ctx = makeQueryCtx();

  std::shared_ptr<TierConfig> seen{std::make_shared<TierConfig>()};
  run(
      [&](exec::Driver& driver) {
        auto operators = driver.operators();
        ASSERT_GE(operators.size(), 1);
        seen = operators[0]->registry<TierConfig>(kTestKey);
      },
      ctx);

  EXPECT_EQ(seen, nullptr);
}

TEST_F(OperatorRegistryTest, typeMismatchOnFallThroughThrows) {
  auto ctx = makeQueryCtx();
  ctx->setRegistry<std::string>(
      kTestKey, std::make_shared<std::string>("value"));

  std::string error;
  run(
      [&](exec::Driver& driver) {
        auto operators = driver.operators();
        ASSERT_GE(operators.size(), 1);
        try {
          operators[0]->registry<TierConfig>(kTestKey);
        } catch (const VeloxException& e) {
          error = e.message();
        }
      },
      ctx);

  EXPECT_NE(
      error.find("Registry type mismatch for key 'testKey'"),
      std::string::npos);
}

TEST_F(OperatorRegistryTest, operatorScopeDuplicateThrowsUnlessOverwrite) {
  auto ctx = makeQueryCtx();
  auto first = std::make_shared<TierConfig>(TierConfig{"first"});
  auto second = std::make_shared<TierConfig>(TierConfig{"second"});

  std::string error;
  std::shared_ptr<TierConfig> afterOverwrite;
  run(
      [&](exec::Driver& driver) {
        auto operators = driver.operators();
        ASSERT_GE(operators.size(), 1);
        operators[0]->setRegistry<TierConfig>(kTestKey, first);
        try {
          operators[0]->setRegistry<TierConfig>(kTestKey, second);
        } catch (const VeloxException& e) {
          error = e.message();
        }
        operators[0]->setRegistry<TierConfig>(
            kTestKey, second, /*overwrite=*/true);
        afterOverwrite = operators[0]->registry<TierConfig>(kTestKey);
      },
      ctx);

  EXPECT_NE(error.find("Registry already set: testKey"), std::string::npos);
  EXPECT_EQ(afterOverwrite, second);
}

} // namespace
} // namespace facebook::velox::exec::test
