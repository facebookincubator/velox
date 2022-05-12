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

#include "velox/substrait/SubstraitToVeloxPlan.h"

namespace facebook::velox::substrait {

/// This class is used to validate whether the computing of
/// a Substrait plan is supported in Velox.
class SubstraitToVeloxPlanValidator {
 public:
  SubstraitToVeloxPlanValidator(core::ExecCtx* execCtx) : execCtx_(execCtx) {}

  /// Validate the Type.
  bool validate(const ::substrait::Type& substraitType);

  /// Validate the Aggregation.
  bool validate(const ::substrait::AggregateRel& aggRel);

  /// Validate the Project.
  bool validate(const ::substrait::ProjectRel& sProject);

  /// Validate the Filter.
  bool validate(const ::substrait::FilterRel& filterRel);

  /// Validate the Read.
  bool validate(const ::substrait::ReadRel& readRel);

  /// Validate the Rel.
  bool validate(const ::substrait::Rel& rel);

  /// Validate the RelRoot.
  bool validate(const ::substrait::RelRoot& root);

  /// Validate the Plan.
  bool validate(const ::substrait::Plan& substraitPlan);

 private:
  /// Used to get types from advanced extension and validate them.
  bool validateInputTypes(
      const ::substrait::extensions::AdvancedExtension& extension,
      std::vector<TypePtr>& types);

  /// An execution context used for function validation.
  core::ExecCtx* execCtx_;

  /// A converter used to convert Substrait plan into Velox's plan node.
  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter_ =
      std::make_shared<SubstraitVeloxPlanConverter>();

  /// A parser used to convert Substrait plan into recognizable representations.
  std::shared_ptr<SubstraitParser> subParser_ =
      std::make_shared<SubstraitParser>();

  /// An expression converter used to convert Substrait representations into
  /// Velox expressions.
  std::shared_ptr<SubstraitVeloxExprConverter> exprConverter_;
};

} // namespace facebook::velox::substrait
