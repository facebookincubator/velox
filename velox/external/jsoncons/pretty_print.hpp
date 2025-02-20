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

#ifndef JSONCONS_PRETTY_PRINT_HPP
#define JSONCONS_PRETTY_PRINT_HPP

#include <cstring>
#include <ostream>
#include <typeinfo>

#include "velox/external/jsoncons/json_encoder.hpp"
#include "velox/external/jsoncons/json_error.hpp"
#include "velox/external/jsoncons/json_options.hpp"
#include "velox/external/jsoncons/json_type_traits.hpp"

namespace facebook::velox::jsoncons {

template <typename Json>
class json_printable
{
public:
    using char_type = typename Json::char_type;

    json_printable(const Json& j, indenting indent)
       : j_(&j), indenting_(indent)
    {
    }

    json_printable(const Json& j,
                   const basic_json_encode_options<char_type>& options,
                   indenting indent)
       : j_(&j), options_(options), indenting_(indent)
    {
    }

    void dump(std::basic_ostream<char_type>& os) const
    {
        j_->dump(os, options_, indenting_);
    }

    friend std::basic_ostream<char_type>& operator<<(std::basic_ostream<char_type>& os, const json_printable<Json>& pr)
    {
        pr.dump(os);
        return os;
    }

    const Json *j_;
    basic_json_encode_options<char_type> options_;
    indenting indenting_;
private:
    json_printable();
};

template <typename Json>
json_printable<Json> print(const Json& j)
{
    return json_printable<Json>(j, indenting::no_indent);
}

template <typename Json>
json_printable<Json> print(const Json& j,
                           const basic_json_encode_options<typename Json::char_type>& options)
{
    return json_printable<Json>(j, options, indenting::no_indent);
}

template <typename Json>
json_printable<Json> pretty_print(const Json& j)
{
    return json_printable<Json>(j, indenting::indent);
}

template <typename Json>
json_printable<Json> pretty_print(const Json& j,
                                  const basic_json_encode_options<typename Json::char_type>& options)
{
    return json_printable<Json>(j, options, indenting::indent);
}

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_PRETTY_PRINT_HPP
