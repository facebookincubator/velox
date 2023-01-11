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

#include "velox/substrait/SubstraitExtensionCollector.h"
#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "velox/substrait/VeloxToSubstraitType.h"
#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/plan.pb.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using SubstraitExprLitStruct = ::substrait::Expression::Literal::Struct;
using SubstraitExprLit = ::substrait::Expression::Literal;
using SubstraitPlan = ::substrait::Plan;
using SubstraitPlanRel = ::substrait::PlanRel;
using SubstraitRelCommon = ::substrait::RelCommon;
using SubstraitRelCommonEmit = ::substrait::RelCommon::Emit;
using SubstraitReadRel = ::substrait::ReadRel;
using SubstraitVirtualTable = ::substrait::ReadRel::VirtualTable;

using facebook::velox::test::VectorTestBase;

namespace facebook::velox::substrait {

using SubstraitExtensionCollectorPtr =
    std::shared_ptr<SubstraitExtensionCollector>;
using VeloxToSubstraitTypeConvertorPtr =
    std::shared_ptr<VeloxToSubstraitTypeConvertor>;

class SubstraitTestPlanBuilder : public VectorTestBase {
 public:
  SubstraitTestPlanBuilder() {
    this->Init();
  };

  void Init() {
    pool_ = memory::getDefaultMemoryPool();
    extensionCollector_ = std::make_shared<SubstraitExtensionCollector>();
    planConverter_ = std::make_shared<SubstraitVeloxPlanConverter>(pool_.get());
  }

  std::shared_ptr<memory::MemoryPool> pool() {
    return pool_;
  }

  std::shared_ptr<SubstraitExtensionCollector> extensionCollector() {
    return extensionCollector_;
  }

  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter() {
    return planConverter_;
  }

  void ValidateReadRelWithEmit(
      const std::vector<RowVectorPtr> data,
      const std::vector<RowVectorPtr>& expected,
      const std::vector<int> emitIndices = {});

 private:
  std::shared_ptr<SubstraitExtensionCollector> extensionCollector_;
  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter_;
  std::shared_ptr<memory::MemoryPool> pool_;
};
} // namespace facebook::velox::substrait
