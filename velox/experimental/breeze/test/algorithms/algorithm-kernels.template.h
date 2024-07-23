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

/*
 * Copyright (c) 2024 by Rivos Inc.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

// Note: This file is simply a pseudo-c++ template for the purposes of being
// specialized across the 6 supported backends. Only a limited subset of what
// you might expect to do in a C++ file will be supported here. In particular:
//    1. Includes will be copied verbatim
//    2. Function signatures will be copied but modified for individual
//    backends
//    3. Function bodies will copied verbatim.
//
// There are a few special attributes for communicating extra information:
//    1. PLATFORM(name) => the function makes use of a backend-specific
//    Platform object called `name`.
//    2. SHARED_MEM(type, name) => the function uses a shared variable of the
//    given type and name.

#define PLATFORM(X) [[clang::annotate("PlatformName=" X)]]
#define SHARED_MEM(T, id) [[clang::annotate("SharedMem=" T ";" id)]]

#include "algorithms/reduce.h"
#include "platforms/platform.h"
#include "utils/types.h"

namespace kernels {

template <typename Op, int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T,
          typename U>
PLATFORM("p")
SHARED_MEM("typename breeze::algorithms::DeviceReduce<PlatformT, U>::Scratch",
           "scratch")
void Reduce(const T* in, U* out, int num_items) {
  breeze::algorithms::DeviceReduce<PlatformT, U>::template Reduce<
      Op, ITEMS_PER_THREAD>(
      p, breeze::utils::make_slice<breeze::utils::GLOBAL>(in),
      breeze::utils::make_slice<breeze::utils::GLOBAL>(out),
      breeze::utils::make_slice<breeze::utils::SHARED>(scratch), num_items);
}

}  // namespace kernels
