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

// GPU shadow for velox/type/SimpleFunctionApi.h.
//
// Mirrors only the type variable tags, decimal parameter aliases, and the
// `Generic<>` wrapper that `Fn<TExec>::call()` bodies and their template
// signatures reference. Drops the heavier function-signature machinery
// (`FunctionSignature`, `SignatureBuilder`, simple-function reflection
// utilities) that lives in the real header but is host-only.
//
// The `getId()` accessors on `TypeVariable` / `IntegerVariable` and the
// deleted-constructor / `static_assert` on `Generic` are omitted because
// they aren't called from `call()` bodies; if a downstream PR needs them,
// they can be added back without functional impact.
#pragma once

#include <cstddef>

namespace facebook::velox {

// A type that can be used in simple function to represent any type.
// Two Generics with the same type variables should bound to the same type.
template <size_t id>
struct TypeVariable {};

using T1 = TypeVariable<1>;
using T2 = TypeVariable<2>;
using T3 = TypeVariable<3>;
using T4 = TypeVariable<4>;

// Integer-valued template parameter (used by ShortDecimal / LongDecimal to
// carry compile-time precision and scale).
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

// Generic represents a polymorphic type in a Velox function signature. The
// concrete instantiation is determined by the type binding rules of the
// signature itself.
template <typename T = AnyType, bool comparable = false, bool orderable = false>
struct Generic {};

using Any = Generic<>;

} // namespace facebook::velox
