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

// CudfLibraryFunctions: wraps cuDF library functions (strings, datetime,
// hashing) as GpuVectorFunction instances. These functions can't be compiled
// through the SFI call() path but cuDF provides optimized GPU implementations.
#pragma once

namespace facebook::velox::gpu {

void registerCudfStringFunctions();
void registerCudfDateTimeFunctions();
void registerCudfHashFunctions();

} // namespace facebook::velox::gpu
