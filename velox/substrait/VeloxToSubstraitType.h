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

#include "velox/core/PlanNode.h"

#include "velox/substrait/proto/substrait/algebra.pb.h"
#include "velox/substrait/proto/substrait/type.pb.h"

namespace facebook::velox::substrait {

class VeloxToSubstraitTypeConvertor {
 public:
  /// Convert Velox RowType to Substrait NamedStruct.
  const ::substrait::NamedStruct& toSubstraitNamedStruct(
      google::protobuf::Arena& arena,
      const velox::RowTypePtr& rowType);

  /// Convert Velox Type to Substrait Type.
  const ::substrait::Type& toSubstraitType(
      google::protobuf::Arena& arena,
      const velox::TypePtr& type);
};

using VeloxToSubstraitTypeConvertorPtr =
    std::shared_ptr<VeloxToSubstraitTypeConvertor>;

} // namespace facebook::velox::substrait
