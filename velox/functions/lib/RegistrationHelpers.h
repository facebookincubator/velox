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

#include "velox/functions/Registerer.h"

namespace facebook::velox::functions {
namespace {

template <template <class> typename T>
void registerBinaryIntegral(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, int8_t, int8_t, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int16_t, int16_t, int16_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int32_t, int32_t, int32_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int64_t, int64_t>(
      aliases, {}, true, defaultOwner);
}

template <template <class> class T, typename TReturn>
void registerBinaryIntegralWithTReturn(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, TReturn, int8_t, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int16_t, int16_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int32_t, int32_t>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int64_t, int64_t>(
      aliases, {}, true, defaultOwner);
}

template <template <class> typename T>
void registerBinaryFloatingPoint(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, double, double, double>(aliases, {}, true, defaultOwner);
  registerFunction<T, float, float, float>(aliases, {}, true, defaultOwner);
}

template <template <class> typename T>
void registerBinaryNumeric(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerBinaryIntegral<T>(aliases, defaultOwner);
  registerBinaryFloatingPoint<T>(aliases, defaultOwner);
}

template <template <class> class T, typename TReturn>
void registerBinaryScalar(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerBinaryIntegralWithTReturn<T, TReturn>(aliases, defaultOwner);
  registerFunction<T, TReturn, double, double>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, float, float>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Varchar, Varchar>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Varbinary, Varbinary>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, bool, bool>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Timestamp, Timestamp>(
      aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, Date, Date>(aliases, {}, true, defaultOwner);
}

template <template <class> class T>
void registerUnaryIntegral(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, int8_t, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int16_t, int16_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int32_t, int32_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, int64_t, int64_t>(aliases, {}, true, defaultOwner);
}

template <template <class> class T, typename TReturn>
void registerUnaryIntegralWithTReturn(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, TReturn, int8_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int16_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int32_t>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, int64_t>(aliases, {}, true, defaultOwner);
}

template <template <class> class T>
void registerUnaryFloatingPoint(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, double, double>(aliases, {}, true, defaultOwner);
  registerFunction<T, float, float>(aliases, {}, true, defaultOwner);
}

template <template <class> class T, typename TReturn>
void registerUnaryFloatingPointWithTReturn(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerFunction<T, TReturn, double>(aliases, {}, true, defaultOwner);
  registerFunction<T, TReturn, float>(aliases, {}, true, defaultOwner);
}

template <template <class> class T>
void registerUnaryNumeric(
    const std::vector<std::string>& aliases,
    std::string_view defaultOwner = {}) {
  registerUnaryIntegral<T>(aliases, defaultOwner);
  registerUnaryFloatingPoint<T>(aliases, defaultOwner);
}

} // namespace

} // namespace facebook::velox::functions
