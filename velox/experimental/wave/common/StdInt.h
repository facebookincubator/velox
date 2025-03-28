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

#ifndef __CUDACC_RTC__
#include <cstdint>
#endif

namespace facebook::velox::wave {

#ifdef __CUDACC_RTC__
using int8_t = char;
using uint8_t = unsigned char;
using int16_t = short;
using uint16_t = unsigned short;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long;
using uint64_t = unsigned long long;
using size_t = unsigned long;
using uintptr_t = unsigned long long;
#else
using int8_t = std::int8_t;
using uint8_t = std::uint8_t;
using int16_t = std::int16_t;
using uint16_t = std::uint16_t;
using int32_t = std::int32_t;
using uint32_t = std::uint32_t;
using int64_t = std::int64_t;
using uint64_t = std::uint64_t;
using size_t = std::size_t;
using uintptr_t = std::uintptr_t;
#endif

} // namespace facebook::velox::wave
