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

// GPU shadow for velox/type/SimpleFunctionApi.h
// Provides type variable tags, decimal parameter aliases, and Generic<>
// used by function signatures.  No Velox core dependencies.
#pragma once

#include <cstddef>

namespace facebook::velox {

template <size_t id>
struct TypeVariable {};

using T1 = TypeVariable<1>;
using T2 = TypeVariable<2>;
using T3 = TypeVariable<3>;
using T4 = TypeVariable<4>;

template <size_t id>
struct IntegerVariable {};

using P1 = IntegerVariable<1>;
using P2 = IntegerVariable<2>;
using P3 = IntegerVariable<3>;
using P4 = IntegerVariable<4>;
using S1 = IntegerVariable<5>;
using S2 = IntegerVariable<6>;
using S3 = IntegerVariable<7>;
using S4 = IntegerVariable<8>;

struct AnyType {};

template <
    typename T = AnyType,
    bool comparable = false,
    bool orderable = false>
struct Generic {};

using Any = Generic<>;

} // namespace facebook::velox
