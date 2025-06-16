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

#include <velox/common/config/Config.h>

namespace facebook::velox4j {

/// Enumeration including the supported presets used for initializing Velox4J.
/// Spark is the only preset we support at this moment.
enum Preset { SPARK = 0 };

extern facebook::velox::config::ConfigBase::Entry<Preset> VELOX4J_INIT_PRESET;
} // namespace facebook::velox4j
