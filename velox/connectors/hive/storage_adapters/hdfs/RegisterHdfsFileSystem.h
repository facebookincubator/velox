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

#include "folly/concurrency/ConcurrentHashMap.h"

namespace facebook::velox::filesystems {

class HdfsFileSystem;

extern folly::ConcurrentHashMap<std::string, std::shared_ptr<HdfsFileSystem>>
    registeredFilesystems;

// Register the HDFS.
void registerHdfsFileSystem();

// Set extra supported schemes for HDFS FileSystem, multiple values separated by commas.
void setHdfsExtraSupportedSchemes(std::string_view schemesStr);

} // namespace facebook::velox::filesystems
