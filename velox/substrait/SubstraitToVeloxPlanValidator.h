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
  SubstraitToVeloxPlanValidator(
      memory::MemoryPool* pool,
      core::ExecCtx* execCtx)
      : pool_(pool), execCtx_(execCtx) {}

  /// Used to validate whether the computing of this Limit is supported.
  bool validate(const ::substrait::FetchRel& fetchRel);

  /// Used to validate whether the computing of this Sort is supported.
  bool validate(const ::substrait::ExpandRel& sExpand);

  /// Used to validate whether the computing of this Sort is supported.
  bool validate(const ::substrait::SortRel& sSort);

  /// Used to validate whether the computing of this Window is supported.
  bool validate(const ::substrait::WindowRel& sWindow);

  /// Used to validate whether the computing of this Aggregation is supported.
  bool validate(const ::substrait::AggregateRel& sAgg);

  /// Used to validate whether the computing of this Project is supported.
  bool validate(const ::substrait::ProjectRel& sProject);

  /// Used to validate whether the computing of this Filter is supported.
  bool validate(const ::substrait::FilterRel& sFilter);

  /// Used to validate Join.
  bool validate(const ::substrait::JoinRel& sJoin);

  /// Used to validate whether the computing of this Read is supported.
  bool validate(const ::substrait::ReadRel& sRead);

  /// Used to validate whether the computing of this Rel is supported.
  bool validate(const ::substrait::Rel& sRel);

  /// Used to validate whether the computing of this RelRoot is supported.
  bool validate(const ::substrait::RelRoot& sRoot);

  /// Used to validate whether the computing of this Plan is supported.
  bool validate(const ::substrait::Plan& sPlan);

 private:
  /// A memory pool used for function validation.
  memory::MemoryPool* pool_;

  /// An execution context used for function validation.
  core::ExecCtx* execCtx_;

  /// A converter used to convert Substrait plan into Velox's plan node.
  std::shared_ptr<SubstraitVeloxPlanConverter> planConverter_ =
      std::make_shared<SubstraitVeloxPlanConverter>(pool_, true);

  /// A parser used to convert Substrait plan into recognizable representations.
  std::shared_ptr<SubstraitParser> subParser_ =
      std::make_shared<SubstraitParser>();

  /// An expression converter used to convert Substrait representations into
  /// Velox expressions.
  std::shared_ptr<SubstraitVeloxExprConverter> exprConverter_;

  /// Used to get types from advanced extension and validate them.
  bool validateInputTypes(
      const ::substrait::extensions::AdvancedExtension& extension,
      std::vector<TypePtr>& types);

  bool validateAggRelFunctionType(const ::substrait::AggregateRel& sAgg);
};

} // namespace facebook::velox::substrait
