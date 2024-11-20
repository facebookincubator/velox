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

#include <memory>
#include <string_view>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace facebook::velox::cudf_velox {

[[nodiscard]] std::shared_ptr<rmm::mr::device_memory_resource>
create_memory_resource(std::string_view mode);

bool cudfDebugEnabled();

} // namespace facebook::velox::cudf_velox
