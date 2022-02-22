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

#include <folly/executors/IOThreadPoolExecutor.h>

#include "SubstraitUtils.h"
#include "velox/common/caching/DataCache.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/dwrf/common/CachedBufferedInput.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/tests/utils/Cursor.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/SumAggregate.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/type/Filter.h"
#include "velox/type/Subfield.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace facebook::velox::substraitconverter {

// This class is used to convert Substrait representations to Velox expressions.
class SubstraitVeloxExprConverter {
 public:
  SubstraitVeloxExprConverter(
      const std::shared_ptr<SubstraitParser>& sub_parser,
      const std::unordered_map<uint64_t, std::string>& functions_map);
  std::shared_ptr<const core::FieldAccessTypedExpr> toVeloxExpr(
      const substrait::Expression::FieldReference& sfield,
      const int32_t& input_plan_node_id);
  std::shared_ptr<const core::ITypedExpr> toVeloxExpr(
      const substrait::Expression::ScalarFunction& sfunc,
      const int32_t& input_plan_node_id);
  std::shared_ptr<const core::ConstantTypedExpr> toVeloxExpr(
      const substrait::Expression::Literal& slit);
  std::shared_ptr<const core::ITypedExpr> toVeloxExpr(const substrait::Expression& sexpr,
                                                      const int32_t& input_plan_node_id);
  int32_t parseReferenceSegment(const substrait::Expression::ReferenceSegment& sref);
  TypePtr getVeloxType(std::string type_name);
  connector::hive::SubfieldFilters toVeloxFilter(
      const std::vector<std::string>& input_name_list,
      const std::vector<TypePtr>& input_type_list, const substrait::Expression& sfilter);

 private:
  void getFlatConditions(
      const substrait::Expression& sfilter,
      std::vector<substrait::Expression_ScalarFunction>* scalar_functions);
  std::shared_ptr<SubstraitParser> sub_parser_;
  std::unordered_map<uint64_t, std::string> functions_map_;
  class FilterInfo;
};

} // namespace facebook::velox::substraitconverter
