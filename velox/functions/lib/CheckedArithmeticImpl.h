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

#include "velox/common/base/CheckedArithmetic.h"

// Names the velox:: overloads, default typeName included, so codegen can keep
// calling functions::checkedPlus(a, b) while code in nested namespaces finds
// the originals without qualification.
namespace facebook::velox::functions {

using facebook::velox::checkedDivide;
using facebook::velox::checkedMinus;
using facebook::velox::checkedModulus;
using facebook::velox::checkedMultiply;
using facebook::velox::checkedNegate;
using facebook::velox::checkedPlus;

} // namespace facebook::velox::functions
