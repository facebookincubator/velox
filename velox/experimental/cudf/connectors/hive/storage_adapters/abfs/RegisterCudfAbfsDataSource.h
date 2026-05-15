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

namespace facebook::velox::cudf_velox::filesystems {

/// Registers `CudfAbfsDataSource` in the cudf datasource registry so
/// that any `abfs://` / `abfss://` URI handed to
/// `getCudfDataSource` is served by the native zero-copy datasource.
/// Mirrors `facebook::velox::filesystems::registerAbfsFileSystem`.
/// Safe to call multiple times; only the first call registers.
/// No-op when `VELOX_ENABLE_ABFS` is not defined.
void registerCudfAbfsDataSource();

} // namespace facebook::velox::cudf_velox::filesystems
