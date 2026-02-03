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
#include "velox/functions/lib/TransformFunctionBase.h"

namespace facebook::velox::functions {
namespace {

// See documentation at https://prestodb.io/docs/current/functions/array.html
class TransformFunction : public TransformFunctionBase {
  // Inherits apply() and signatures() from base class.
  // No additional lambda arguments needed for Presto.
};

} // namespace

/// transform is null preserving for the array. But since an
/// expr tree with a lambda depends on all named fields, including
/// captures, a null in a capture does not automatically make a
/// null result.

VELOX_DECLARE_VECTOR_FUNCTION_WITH_METADATA(
    udf_transform,
    TransformFunction::signatures(),
    exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
    std::make_unique<TransformFunction>());

} // namespace facebook::velox::functions
