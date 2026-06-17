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

namespace facebook::velox::cxl {

/// Installs a DriverAdapter that, for a query carrying a CXL tier, swaps
/// operators for their CXL variants. The adapter checks the CXL tier once, then
/// tries each known per-operator replacement. Call once at startup.
void registerCxlDriverAdapter();

} // namespace facebook::velox::cxl
