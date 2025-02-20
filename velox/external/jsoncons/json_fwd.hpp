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

#ifndef JSONCONS_JSON_FWD_HPP
#define JSONCONS_JSON_FWD_HPP

#include <memory> // std::allocator

namespace facebook::velox::jsoncons {

struct sorted_policy;
                        
template <typename CharT, 
          typename Policy = sorted_policy, 
          typename Allocator = std::allocator<CharT>>
class basic_json;

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_JSON_FWD_HPP
