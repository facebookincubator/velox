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

#ifndef JSONCONS_ALLOCATOR_HOLDER_HPP
#define JSONCONS_ALLOCATOR_HOLDER_HPP

namespace facebook::velox::jsoncons {

template <typename Allocator>
class allocator_holder
{
public:
    using allocator_type = Allocator;
private:
    allocator_type alloc_;
public:
    allocator_holder() = default;
    allocator_holder(const allocator_holder&)  = default;
    allocator_holder(allocator_holder&&)  = default;
    allocator_holder& operator=(const allocator_holder&) = delete;
    allocator_holder(const allocator_type& alloc)
        : alloc_(alloc)
        {}
    ~allocator_holder() = default;
    
    allocator_type get_allocator() const
    {
        return alloc_;
    }
};

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_ALLOCATOR_HOLDER_HPP
