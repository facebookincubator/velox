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


#include "folly/IPAddress.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox::functions {

template <typename T>
struct IPPrefixFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  FOLLY_ALWAYS_INLINE bool call(out_type<Varchar>& result,
                                const arg_type<Varchar>& ip ,
                                const uint8_t prefixBits) {

    const folly::IPAddress ipAddress = folly::IPAddress(ip);
    auto ipResult = ipAddress.mask(prefixBits);
    result.setNoCopy(StringView(ipResult.str()));

    return true;
  }
};

} // namespace facebook::velox::functions
