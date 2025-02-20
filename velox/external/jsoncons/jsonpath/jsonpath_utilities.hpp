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

#ifndef JSONCONS_EXT_JSONPATH_JSONPATH_UTILITIES_HPP
#define JSONCONS_EXT_JSONPATH_JSONPATH_UTILITIES_HPP

#include <cstddef>

namespace facebook::velox::jsoncons { namespace jsonpath {

    template <typename CharT,typename Sink>
    std::size_t escape_string(const CharT* s, std::size_t length, Sink& sink)
    {
        std::size_t count = 0;
        const CharT* begin = s;
        const CharT* end = s + length;
        for (const CharT* it = begin; it != end; ++it)
        {
            CharT c = *it;
            switch (c)
            {
                case '\\':
                    sink.push_back('\\');
                    sink.push_back('\\');
                    count += 2;
                    break;
                case '\'':
                    sink.push_back('\\');
                    sink.push_back('\'');
                    count += 2;
                    break;
                case '\b':
                    sink.push_back('\\');
                    sink.push_back('b');
                    count += 2;
                    break;
                case '\f':
                    sink.push_back('\\');
                    sink.push_back('f');
                    count += 2;
                    break;
                case '\n':
                    sink.push_back('\\');
                    sink.push_back('n');
                    count += 2;
                    break;
                case '\r':
                    sink.push_back('\\');
                    sink.push_back('r');
                    count += 2;
                    break;
                case '\t':
                    sink.push_back('\\');
                    sink.push_back('t');
                    count += 2;
                    break;
                default:
                    sink.push_back(c);
                    ++count;
                    break;
            }
        }
        return count;
    }
    
} // namespace jsonpath
} // namespace facebook::velox::jsoncons

#endif // JSONCONS_EXT_JSONPATH_JSONPATH_UTILITIES_HPP
