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

// Windows build compatibility shim.
//
// Velox calls folly::available_concurrency(), a cgroup-aware helper that was
// added to folly after the folly version pinned by the Windows vcpkg manifest.
// The pinned folly only provides folly::hardware_concurrency(). Windows has no
// cgroup CPU accounting, so hardware_concurrency() is an exact substitute.
//
// This header is only included on Windows; POSIX builds use folly's real
// available_concurrency() and never see this shim.
#ifdef _WIN32
#include <folly/system/HardwareConcurrency.h>

namespace folly {
inline unsigned int available_concurrency() noexcept {
  return hardware_concurrency();
}
} // namespace folly
#endif // _WIN32
