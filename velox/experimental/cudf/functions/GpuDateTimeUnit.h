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

#include "velox/functions/lib/DateTimeUnit.h"

#include <optional>
#include <string_view>

/// Names only POD and standard types, so it is safe on both sides of the
/// shadow include path.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// functions::fromDateTimeUnitString() over a plain string, for the shadow
/// side, whose own fromDateTimeUnitString() is the device form. Defined on the
/// host.
std::optional<functions::DateTimeUnit> gpuFromDateTimeUnitString(
    std::string_view unitString,
    bool throwIfInvalid,
    bool allowMicro,
    bool allowAbbreviated);

} // namespace facebook::velox::cudf_velox::gpu_sfi
