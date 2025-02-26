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
// Copyright 2013-2025 Daniel Parker
// Distributed under the Boost license, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// See https://github.com/danielaparker/jsoncons for latest version

#ifndef JSONCONS_DETAIL_ENDIAN_HPP
#define JSONCONS_DETAIL_ENDIAN_HPP

#if defined(__sun)
#  include <sys/byteorder.h>
#endif

namespace facebook::velox::jsoncons { 
namespace detail {

    enum class endian
    {
    #if defined(_MSC_VER) 
    // MSVC, which implies Windows, which implies little-endian
         little = 0,
         big    = 1,
         native = little
    #elif defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) && defined(__BYTE_ORDER__) 
         little = __ORDER_LITTLE_ENDIAN__,
         big    = __ORDER_BIG_ENDIAN__,
         native = __BYTE_ORDER__
    #elif defined(_BIG_ENDIAN) && !defined(_LITTLE_ENDIAN)
        little = 0,
        big    = 1,
        native = big
    #elif !defined(_BIG_ENDIAN) && defined(_LITTLE_ENDIAN)
        little = 0,
        big    = 1,
        native = little
    #else
    #error "Unable to determine byte order!"
    #endif
    };

} // namespace detail
} // namespace facebook::velox::jsoncons

#endif // JSONCONS_DETAIL_ENDIAN_HPP
