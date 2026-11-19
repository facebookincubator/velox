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
#include "velox/functions/lib/ArrayContains.h"

namespace facebook::velox::functions {

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_contains,
    arrayContainsSignatures(),
    makeArrayContains(/*throwOnNestedNull=*/true));

// Internal function only used for testing. This function allows the array to
// have null elements and considers null as a value, i.e., null == null.
VELOX_DECLARE_VECTOR_FUNCTION(
    udf_$internal$contains,
    arrayContainsSignatures(),
    makeArrayContains(/*throwOnNestedNull=*/false));

} // namespace facebook::velox::functions
