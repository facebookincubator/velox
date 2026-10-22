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
#include <string>

namespace facebook::velox::functions::prestosql {
void registerArithmeticFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerCheckedArithmeticFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerComparisonFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerArrayFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerInternalFunctions(std::string_view defaultOwner = {});

void registerMapFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerJsonFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerHyperLogFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerKHyperLogLogFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerTDigestFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerQDigestFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerSfmSketchFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerBingTileFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerEnumFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerGeneralFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerDateTimeFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerURLFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerStringFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerBinaryFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerBitwiseFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerAllScalarFunctions(
    const std::string& prefix = "",
    std::string_view defaultOwner = {});

void registerMapAllowingDuplicates(
    const std::string& name,
    const std::string& prefix = "",
    std::string_view defaultOwner = {});
} // namespace facebook::velox::functions::prestosql
