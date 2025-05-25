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

#ifndef JSONCONS_SER_CONTEXT_HPP
#define JSONCONS_SER_CONTEXT_HPP

#include <cstddef>

namespace facebook::velox::jsoncons {

class ser_context
{
public:
    virtual ~ser_context() = default;

    virtual size_t line() const
    {
        return 0;
    }

    virtual size_t column() const
    {
        return 0;
    }

    virtual size_t position() const
    {
        return 0;
    }

    virtual size_t end_position() const
    {
        return 0;
    }
};

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_SER_CONTEXT_HPP
