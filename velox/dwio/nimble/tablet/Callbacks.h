/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <functional>
#include <memory>

#include "velox/common/file/Region.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

/// Callback type for loading metadata from a MetadataSection.
using LoadMetadataFn =
    std::function<std::unique_ptr<MetadataBuffer>(const MetadataSection&)>;

/// Callback type for loading data from a file region.
/// Returns a SeekableInputStream positioned at the start of the region.
using LoadDataFn =
    std::function<std::unique_ptr<velox::dwio::common::SeekableInputStream>(
        const velox::common::Region&)>;

} // namespace facebook::nimble
