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

#include "velox/common/memory/windows/PosixMemoryCompat.h"

// Note: All static members now use Meyers Singleton pattern (function-local statics)
// to avoid static initialization order fiasco. No explicit static member definitions needed.

namespace facebook::velox::memory::windows {
// Intentionally empty - all initialization happens via function-local statics
} // namespace facebook::velox::memory::windows

