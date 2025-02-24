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

#ifndef JSONCONS_JSON_FILTER_HPP
#define JSONCONS_JSON_FILTER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>

#include "velox/external/jsoncons/config/compiler_support.hpp"
#include "velox/external/jsoncons/utility/byte_string.hpp"
#include "velox/external/jsoncons/json_exception.hpp"
#include "velox/external/jsoncons/json_visitor.hpp"
#include "velox/external/jsoncons/ser_context.hpp"
#include "velox/external/jsoncons/tag_type.hpp"
#include "velox/external/jsoncons/utility/unicode_traits.hpp"

namespace facebook::velox::jsoncons {

template <typename CharT>
class basic_json_filter : public basic_json_visitor<CharT>
{
public:
    using typename basic_json_visitor<CharT>::char_type;
    using typename basic_json_visitor<CharT>::string_view_type;
private:
    basic_json_visitor<char_type>* destination_;

public:
    basic_json_filter(basic_json_visitor<char_type>& visitor)
        : destination_(std::addressof(visitor))
    {
    }

    basic_json_visitor<char_type>& destination()
    {
        return *destination_;
    }

private:
    void visit_flush() override
    {
        destination_->flush();
    }

    bool visit_begin_object(semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->begin_object(tag, context, ec);
    }

    bool visit_begin_object(std::size_t length, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->begin_object(length, tag, context, ec);
    }

    bool visit_end_object(const ser_context& context, std::error_code& ec) override
    {
        return destination_->end_object(context, ec);
    }

    bool visit_begin_array(semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->begin_array(tag, context, ec);
    }

    bool visit_begin_array(std::size_t length, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->begin_array(length, tag, context, ec);
    }

    bool visit_end_array(const ser_context& context, std::error_code& ec) override
    {
        return destination_->end_array(context, ec);
    }

    bool visit_key(const string_view_type& name,
                 const ser_context& context,
                 std::error_code& ec) override
    {
        return destination_->key(name, context, ec);
    }

    bool visit_string(const string_view_type& value,
                      semantic_tag tag,
                      const ser_context& context,
                      std::error_code& ec) override
    {
        return destination_->string_value(value, tag, context, ec);
    }

    bool visit_byte_string(const byte_string_view& b, 
                           semantic_tag tag,
                           const ser_context& context,
                           std::error_code& ec) override
    {
        return destination_->byte_string_value(b, tag, context, ec);
    }

    bool visit_byte_string(const byte_string_view& b, 
                           uint64_t ext_tag,
                           const ser_context& context,
                           std::error_code& ec) override
    {
        return destination_->byte_string_value(b, ext_tag, context, ec);
    }

    bool visit_uint64(uint64_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->uint64_value(value, tag, context, ec);
    }

    bool visit_int64(int64_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->int64_value(value, tag, context, ec);
    }

    bool visit_half(uint16_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->half_value(value, tag, context, ec);
    }

    bool visit_double(double value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->double_value(value, tag, context, ec);
    }

    bool visit_bool(bool value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->bool_value(value, tag, context, ec);
    }

    bool visit_null(semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->null_value(tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint8_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint32_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint64_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int8_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int32_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int64_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(half_arg_t, 
                        const jsoncons::span<const uint16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(half_arg, s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const float>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const double>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_begin_multi_dim(const jsoncons::span<const size_t>& shape,
                            semantic_tag tag,
                            const ser_context& context, 
                            std::error_code& ec) override
    {
        return destination_->begin_multi_dim(shape, tag, context, ec);
    }

    bool visit_end_multi_dim(const ser_context& context,
                          std::error_code& ec) override
    {
        return destination_->end_multi_dim(context, ec);
    }
};

template <typename CharT>
class basic_rename_object_key_filter : public basic_json_filter<CharT>
{
public:
    using typename basic_json_filter<CharT>::string_view_type;

private:
    std::basic_string<CharT> name_;
    std::basic_string<CharT> new_name_;
public:
    basic_rename_object_key_filter(const std::basic_string<CharT>& name,
                             const std::basic_string<CharT>& new_name,
                             basic_json_visitor<CharT>& visitor)
        : basic_json_filter<CharT>(visitor), 
          name_(name), new_name_(new_name)
    {
    }

private:
    bool visit_key(const string_view_type& name,
                 const ser_context& context,
                 std::error_code& ec) override
    {
        if (name == name_)
        {
            return this->destination().key(new_name_,context, ec);
        }
        else
        {
            return this->destination().key(name,context,ec);
        }
    }
};

template <typename From,typename To>
class json_visitor_adaptor_base : public From
{
public:
    using typename From::string_view_type;
private:
    To* destination_;
public:

    // noncopyable
    json_visitor_adaptor_base(const json_visitor_adaptor_base&) = delete;

    json_visitor_adaptor_base(To& visitor)
        : destination_(std::addressof(visitor))
    {
    }

    // moveable
    json_visitor_adaptor_base(json_visitor_adaptor_base&&) = default;

    json_visitor_adaptor_base& operator=(const json_visitor_adaptor_base&) = delete;
    json_visitor_adaptor_base& operator=(json_visitor_adaptor_base&&) = default;

    To& destination()
    {
        return *destination_;
    }

private:
    void visit_flush() override
    {
        destination_->flush();
    }

    bool visit_begin_object(semantic_tag tag, 
                         const ser_context& context,
                         std::error_code& ec) override
    {
        return destination_->begin_object(tag, context, ec);
    }

    bool visit_begin_object(std::size_t length, 
                         semantic_tag tag, 
                         const ser_context& context,
                         std::error_code& ec) override
    {
        return destination_->begin_object(length, tag, context, ec);
    }

    bool visit_end_object(const ser_context& context, std::error_code& ec) override
    {
        return destination_->end_object(context, ec);
    }

    bool visit_begin_array(semantic_tag tag, 
                        const ser_context& context,
                        std::error_code& ec) override
    {
        return destination_->begin_array(tag, context, ec);
    }

    bool visit_begin_array(std::size_t length, 
                        semantic_tag tag, 
                        const ser_context& context,
                        std::error_code& ec) override
    {
        return destination_->begin_array(length, tag, context, ec);
    }

    bool visit_end_array(const ser_context& context, std::error_code& ec) override
    {
        return destination_->end_array(context, ec);
    }

    bool visit_byte_string(const byte_string_view& b, 
                           semantic_tag tag,
                           const ser_context& context,
                           std::error_code& ec) override
    {
        return destination_->byte_string_value(b, tag, context, ec);
    }

    bool visit_byte_string(const byte_string_view& b, 
                           uint64_t ext_tag,
                           const ser_context& context,
                           std::error_code& ec) override
    {
        return destination_->byte_string_value(b, ext_tag, context, ec);
    }

    bool visit_half(uint16_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->half_value(value, tag, context, ec);
    }

    bool visit_double(double value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->double_value(value, tag, context, ec);
    }

    bool visit_int64(int64_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->int64_value(value, tag, context, ec);
    }

    bool visit_uint64(uint64_t value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->uint64_value(value, tag, context, ec);
    }

    bool visit_bool(bool value, semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->bool_value(value, tag, context, ec);
    }

    bool visit_null(semantic_tag tag, const ser_context& context, std::error_code& ec) override
    {
        return destination_->null_value(tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint8_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint32_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const uint64_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int8_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int32_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const int64_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(half_arg_t, 
                        const jsoncons::span<const uint16_t>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(half_arg, s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const float>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_typed_array(const jsoncons::span<const double>& s, 
                        semantic_tag tag,
                        const ser_context& context, 
                        std::error_code& ec) override
    {
        return destination_->typed_array(s, tag, context, ec);
    }

    bool visit_begin_multi_dim(const jsoncons::span<const size_t>& shape,
                            semantic_tag tag,
                            const ser_context& context, 
                            std::error_code& ec) override
    {
        return destination_->begin_multi_dim(shape, tag, context, ec);
    }

    bool visit_end_multi_dim(const ser_context& context,
                          std::error_code& ec) override
    {
        return destination_->end_multi_dim(context, ec);
    }

};

template <typename From,typename To,typename Enable=void>
class json_visitor_adaptor 
{
};

template <typename From,typename To>
class json_visitor_adaptor<From,To,typename std::enable_if<extension_traits::is_narrow_character<typename From::char_type>::value &&
                                                           extension_traits::is_narrow_character<typename To::char_type>::value>::type> : public json_visitor_adaptor_base<From,To>
{
    using supertype = json_visitor_adaptor_base<From,To>;
    using to_char_type = typename To::char_type;
    using from_char_type = typename From::char_type;
public:
    using typename From::string_view_type;
    using supertype::destination;
public:

    // noncopyable
    json_visitor_adaptor(const json_visitor_adaptor&) = delete;
    // moveable
    json_visitor_adaptor(json_visitor_adaptor&&) = default;

    json_visitor_adaptor(To& visitor)
        : supertype(visitor)
    {
    }

    json_visitor_adaptor& operator=(const json_visitor_adaptor&) = delete;
    json_visitor_adaptor& operator=(json_visitor_adaptor&&) = default;

private:

    bool visit_key(const string_view_type& key,
                 const ser_context& context,
                 std::error_code& ec) override
    {
        return destination().key(string_view_type(reinterpret_cast<const to_char_type*>(key.data()),key.size()), context, ec);
    }

    bool visit_string(const string_view_type& value,
                      semantic_tag tag,
                      const ser_context& context,
                      std::error_code& ec) override
    {
        return destination().string_value(string_view_type(reinterpret_cast<const to_char_type*>(value.data()),value.size()), tag, context, ec);
    }
};

template <typename From,typename To>
class json_visitor_adaptor<From,To,typename std::enable_if<!(extension_traits::is_narrow_character<typename From::char_type>::value &&
                                                             extension_traits::is_narrow_character<typename To::char_type>::value)>::type> : public json_visitor_adaptor_base<From,To>
{
    using supertype = json_visitor_adaptor_base<From,To>;
public:
    using typename From::string_view_type;
    using supertype::destination;
public:

    // noncopyable
    json_visitor_adaptor(const json_visitor_adaptor&) = delete;
    // moveable
    json_visitor_adaptor(json_visitor_adaptor&&) = default;

    json_visitor_adaptor(To& visitor)
        : supertype(visitor)
    {
    }

    json_visitor_adaptor& operator=(const json_visitor_adaptor&) = delete;
    json_visitor_adaptor& operator=(json_visitor_adaptor&&) = default;

private:

    bool visit_key(const string_view_type& name,
                 const ser_context& context,
                 std::error_code& ec) override
    {
        std::basic_string<typename To::char_type> target;
        auto result = unicode_traits::convert(name.data(), name.size(), target, unicode_traits::conv_flags::strict);
        if (result.ec != unicode_traits::conv_errc())
        {
            ec = result.ec;
        }
        return destination().key(target, context, ec);
    }

    bool visit_string(const string_view_type& value,
                      semantic_tag tag,
                      const ser_context& context,
                      std::error_code& ec) override
    {
        std::basic_string<typename To::char_type> target;
        auto result = unicode_traits::convert(value.data(), value.size(),
                                              target,unicode_traits::conv_flags::strict);
        if (result.ec != unicode_traits::conv_errc())
        {
            JSONCONS_THROW(ser_error(result.ec));
        }
        return destination().string_value(target, tag, context, ec);
    }
};

template <typename From,typename To>
json_visitor_adaptor<From,To> make_json_visitor_adaptor(To& to)
{
    return json_visitor_adaptor<From, To>(to);
}

using json_filter = basic_json_filter<char>;
using wjson_filter = basic_json_filter<wchar_t>;
using rename_object_key_filter = basic_rename_object_key_filter<char>;
using wrename_object_key_filter = basic_rename_object_key_filter<wchar_t>;

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_JSON_FILTER_HPP
