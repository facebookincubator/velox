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

// GPU shadow for velox/core/Metaprogramming.h
// Provides DECLARE_METHOD_RESOLVER and DECLARE_CONDITIONAL_TYPE_NAME
// macros without pulling in the full Velox metaprogramming utilities.
#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace facebook::velox::util {

#define DECLARE_METHOD_RESOLVER(Name, MethodName)              \
  struct Name {                                                \
    template <class __T, typename... __TArgs>                  \
    constexpr auto resolve(__TArgs&&... args) const            \
        -> decltype(std::declval<__T>().MethodName(args...)) { \
      return {};                                               \
    }                                                          \
  }

#define DECLARE_CONDITIONAL_TYPE_NAME(Name, TypeName, OtherTypeName) \
  struct Name {                                                      \
    template <typename __T, typename = void>                         \
    struct resolve {                                                 \
      using type = typename __T::OtherTypeName;                      \
    };                                                               \
                                                                     \
    template <typename __T>                                          \
    struct resolve<                                                  \
        __T,                                                         \
        std::void_t<decltype(sizeof(typename __T::TypeName))>> {     \
      using type = typename __T::TypeName;                           \
    };                                                               \
  }

template <typename C, class TResolver, typename TRet, typename... TArgs>
struct has_method {
 private:
  template <typename T>
  static constexpr auto check(T*) -> typename std::is_same<
      decltype(std::declval<TResolver>().template resolve<T>(
          std::declval<TArgs>()...)),
      TRet>::type {
    return {};
  }

  template <typename>
  static constexpr std::false_type check(...) {
    return std::false_type();
  }

  using type = decltype(check<C>(nullptr));

 public:
  static constexpr bool value = type::value;
};

} // namespace facebook::velox::util
