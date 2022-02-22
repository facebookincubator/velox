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
#include "velox/common/base/test_utils/GTestUtils.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Type.h"
#include "velox/type/tests/FilterBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include "velox/substrait_converter/SubstraitToVeloxPlan.h"

#include <fstream>
#include <iostream>

#if __has_include("filesystem")
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::exec;
using namespace facebook::velox::common::test;
using namespace facebook::velox::exec::test;

class PlanConversionTest : public virtual HiveConnectorTestBase,
                           public testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    useAsyncCache_ = GetParam();
    HiveConnectorTestBase::SetUp();
  }

  static void SetUpTestCase() {
    HiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const std::shared_ptr<const RowType>& rowType) {
    return HiveConnectorTestBase::makeVectors(rowType, count, rowsPerVector);
  }

  class VeloxConverter {
   public:
    VeloxConverter() {}

    class WholeComputeResultIterator {
     public:
      WholeComputeResultIterator(
          const std::shared_ptr<const core::PlanNode>& plan_node,
          const u_int32_t& index,
          const std::vector<std::string>& paths,
          const std::vector<u_int64_t>& starts,
          const std::vector<u_int64_t>& lengths)
          : plan_node_(plan_node),
            index_(index),
            paths_(paths),
            starts_(starts),
            lengths_(lengths) {
        std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
            connectorSplits;
        for (int idx = 0; idx < paths.size(); idx++) {
          auto path = paths[idx];
          auto start = starts[idx];
          auto length = lengths[idx];
          auto split = std::make_shared<
              facebook::velox::connector::hive::HiveConnectorSplit>(
              "hive-connector",
              path,
              facebook::velox::dwio::common::FileFormat::ORC,
              start,
              length);
          connectorSplits.push_back(split);
        }
        splits_.reserve(connectorSplits.size());
        for (const auto& connectorSplit : connectorSplits) {
          splits_.emplace_back(exec::Split(folly::copy(connectorSplit), -1));
        }
        params_.planNode = plan_node;
        cursor_ = std::make_unique<exec::test::TaskCursor>(params_);
        addSplits_ = [&](Task* task) {
          if (noMoreSplits_) {
            return;
          }
          for (auto& split : splits_) {
            task->addSplit("0", std::move(split));
          }
          task->noMoreSplits("0");
          noMoreSplits_ = true;
        };
      }

      bool HasNext() {
        if (!may_has_next_) {
          return false;
        }
        if (num_rows_ > 0) {
          return true;
        } else {
          addSplits_(cursor_->task().get());
          if (cursor_->moveNext()) {
            result_ = cursor_->current();
            num_rows_ += result_->size();
            return true;
          } else {
            may_has_next_ = false;
            return false;
          }
        }
      }

      RowVectorPtr Next() {
        num_rows_ = 0;
        return result_;
      }

     private:
      const std::shared_ptr<const core::PlanNode> plan_node_;
      std::unique_ptr<exec::test::TaskCursor> cursor_;
      exec::test::CursorParameters params_;
      std::vector<exec::Split> splits_;
      bool noMoreSplits_ = false;
      std::function<void(exec::Task*)> addSplits_;
      u_int32_t index_;
      std::vector<std::string> paths_;
      std::vector<u_int64_t> starts_;
      std::vector<u_int64_t> lengths_;
      uint64_t num_rows_ = 0;
      bool may_has_next_ = true;
      RowVectorPtr result_;
    };

    std::shared_ptr<WholeComputeResultIterator> getResIter(
        const std::string& substrait_plan_path) {
      std::fstream subData(
          substrait_plan_path, std::ios::binary | std::ios::in);
      substrait::Plan subPlan;
      subPlan.ParseFromIstream(&subData);
      auto planConverter = std::make_shared<
          facebook::velox::substraitconverter::SubstraitVeloxPlanConverter>();
      auto planNode = planConverter->toVeloxPlan(subPlan);
      auto resIter = std::make_shared<WholeComputeResultIterator>(
          planNode,
          planConverter->getPartitionIndex(),
          planConverter->getPaths(),
          planConverter->getStarts(),
          planConverter->getLengths());
      return resIter;
    }
  };
};

TEST_P(PlanConversionTest, queryTest) {
  std::string subPlanPath = "";
  auto veloxConverter = std::make_shared<VeloxConverter>();
  auto resIter = veloxConverter->getResIter(subPlanPath);
  while (resIter->HasNext()) {
    auto rv = resIter->Next();
  }
}
