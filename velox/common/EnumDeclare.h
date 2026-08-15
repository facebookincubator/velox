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

#include <iosfwd>
#include <optional>
#include <string_view>

/// Lightweight declaration macros for enum-to-name mappings.
///
/// Include this header in .h files. Include EnumDefine.h only in .cpp files
/// that use the DEFINE macros. EnumDefine.h pulls in heavy transitive includes
/// (folly/F14Map.h, Exceptions.h, ~8M preprocessed) that propagate to every
/// file that includes your header.
///
/// Usage:
///
/// In the header file, define the enum and declare its name mapping:
///
/// #include "velox/common/EnumDeclare.h"
///
/// enum class Foo {...};
///
/// VELOX_DECLARE_ENUM_NAME(Foo);
///
/// In the .cpp file, define the mapping:
///
/// #include "velox/common/EnumDefine.h"
///
/// namespace {
/// const auto& fooNames() {
///   static const folly::F14FastMap<Foo, std::string_view> kNames = {
///       {Foo::kFirst, "FIRST"},
///       {Foo::kSecond, "SECOND"},
///        ...
///   };
///   return kNames;
/// }
/// } // namespace
///
/// VELOX_DEFINE_ENUM_NAME(Foo, fooNames);
///
/// In client code, use FooName::toName(Foo::kFirst) to get the name of the
/// enum and FooName::toFoo("FIRST") or FooName::tryToFoo("FIRST") to get the
/// enum value. toFoo throws an exception if the input is not a valid name,
/// while tryToFoo returns std::nullopt.
///
/// Use _EMBEDDED_ versions of the macros to define enums embedded in other
/// classes.

#define VELOX_DECLARE_ENUM_NAME(EnumType)                                  \
  struct EnumType##Name {                                                  \
    static std::string_view toName(EnumType value);                        \
    static EnumType to##EnumType(std::string_view name);                   \
    static std::optional<EnumType> tryTo##EnumType(std::string_view name); \
  };                                                                       \
  std::ostream& operator<<(std::ostream& os, const EnumType& value);

#define VELOX_DECLARE_EMBEDDED_ENUM_NAME(EnumType)     \
  static std::string_view toName(EnumType value);      \
  static EnumType to##EnumType(std::string_view name); \
  static std::optional<EnumType> tryTo##EnumType(std::string_view name);
