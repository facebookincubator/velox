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

/// Defines an fmt formatter that prints an enum as its name. Write it in the
/// global namespace after VELOX_DECLARE_ENUM_NAME, with the enum's fully
/// qualified name:
///
/// VELOX_ENUM_FORMATTER(facebook::velox::Foo);
///
/// Use VELOX_EMBEDDED_ENUM_FORMATTER for an enum embedded in a class:
///
/// VELOX_EMBEDDED_ENUM_FORMATTER(facebook::velox::Bar, Kind);
///
/// The file that uses either macro must include fmt/format.h.

#define VELOX_ENUM_FORMATTER(enum)                                   \
  template <>                                                        \
  struct fmt::formatter<::enum> : fmt::formatter<std::string_view> { \
    template <typename FormatContext>                                \
    auto format(::enum value, FormatContext& ctx) const {            \
      return fmt::formatter<std::string_view>::format(               \
          ::enum##Name::toName(value), ctx);                         \
    }                                                                \
  }

#define VELOX_EMBEDDED_ENUM_FORMATTER(class, enum)                           \
  template <>                                                                \
  struct fmt::formatter<::class ::enum> : fmt::formatter<std::string_view> { \
    template <typename FormatContext>                                        \
    auto format(::class ::enum value, FormatContext& ctx) const {            \
      return fmt::formatter<std::string_view>::format(                       \
          ::class ::toName(value), ctx);                                     \
    }                                                                        \
  }
