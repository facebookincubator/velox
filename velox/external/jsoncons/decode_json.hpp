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

#ifndef JSONCONS_DECODE_JSON_HPP
#define JSONCONS_DECODE_JSON_HPP

#include <iostream>
#include <istream> // std::basic_istream
#include <tuple>

#include "velox/external/jsoncons/config/compiler_support.hpp"
#include "velox/external/jsoncons/allocator_set.hpp"
#include "velox/external/jsoncons/conv_error.hpp"
#include "velox/external/jsoncons/decode_traits.hpp"
#include "velox/external/jsoncons/json_cursor.hpp"
#include "velox/external/jsoncons/basic_json.hpp"
#include "velox/external/jsoncons/source.hpp"

namespace facebook::velox::jsoncons {

    // decode_json

    template <typename T,typename Source>
    typename std::enable_if<extension_traits::is_basic_json<T>::value &&
                            extension_traits::is_sequence_of<Source,typename T::char_type>::value,T>::type
    decode_json(const Source& s,
                const basic_json_decode_options<typename Source::value_type>& options = basic_json_decode_options<typename Source::value_type>())
    {
        using char_type = typename Source::value_type;

        jsoncons::json_decoder<T> decoder;
        basic_json_reader<char_type, string_source<char_type>> reader(s, decoder, options);
        reader.read();
        if (!decoder.is_valid())
        {
            JSONCONS_THROW(ser_error(conv_errc::conversion_failed, reader.line(), reader.column()));
        }
        return decoder.get_result();
    }

    template <typename T,typename Source>
    typename std::enable_if<!extension_traits::is_basic_json<T>::value &&
                            extension_traits::is_char_sequence<Source>::value,T>::type
    decode_json(const Source& s,
                const basic_json_decode_options<typename Source::value_type>& options = basic_json_decode_options<typename Source::value_type>())
    {
        using char_type = typename Source::value_type;

        basic_json_cursor<char_type,string_source<char_type>> cursor(s, options, default_json_parsing());
        jsoncons::json_decoder<basic_json<char_type>> decoder;
        std::error_code ec;
        T val = decode_traits<T,char_type>::decode(cursor, decoder, ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec, cursor.context().line(), cursor.context().column()));
        }
        return val;
    }

    template <typename T,typename CharT>
    typename std::enable_if<extension_traits::is_basic_json<T>::value,T>::type
    decode_json(std::basic_istream<CharT>& is,
                const basic_json_decode_options<CharT>& options = basic_json_decode_options<CharT>())
    {
        jsoncons::json_decoder<T> decoder;
        basic_json_reader<CharT, stream_source<CharT>> reader(is, decoder, options);
        reader.read();
        if (!decoder.is_valid())
        {
            JSONCONS_THROW(ser_error(conv_errc::conversion_failed, reader.line(), reader.column()));
        }
        return decoder.get_result();
    }

    template <typename T,typename CharT>
    typename std::enable_if<!extension_traits::is_basic_json<T>::value,T>::type
    decode_json(std::basic_istream<CharT>& is,
                const basic_json_decode_options<CharT>& options = basic_json_decode_options<CharT>())
    {
        basic_json_cursor<CharT> cursor(is, options, default_json_parsing());
        json_decoder<basic_json<CharT>> decoder{};

        std::error_code ec;
        T val = decode_traits<T,CharT>::decode(cursor, decoder, ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec, cursor.line(), cursor.column()));
        }
        return val;
    }

    template <typename T,typename InputIt>
    typename std::enable_if<extension_traits::is_basic_json<T>::value,T>::type
    decode_json(InputIt first, InputIt last,
                const basic_json_decode_options<typename std::iterator_traits<InputIt>::value_type>& options = 
                    basic_json_decode_options<typename std::iterator_traits<InputIt>::value_type>())
    {
        using char_type = typename std::iterator_traits<InputIt>::value_type;

        jsoncons::json_decoder<T> decoder;
        basic_json_reader<char_type, iterator_source<InputIt>> reader(iterator_source<InputIt>(first,last), decoder, options);
        reader.read();
        if (!decoder.is_valid())
        {
            JSONCONS_THROW(ser_error(conv_errc::conversion_failed, reader.line(), reader.column()));
        }
        return decoder.get_result();
    }

    template <typename T,typename InputIt>
    typename std::enable_if<!extension_traits::is_basic_json<T>::value,T>::type
    decode_json(InputIt first, InputIt last,
                const basic_json_decode_options<typename std::iterator_traits<InputIt>::value_type>& options = 
                    basic_json_decode_options<typename std::iterator_traits<InputIt>::value_type>())
    {
        using char_type = typename std::iterator_traits<InputIt>::value_type;

        basic_json_cursor<char_type,iterator_source<InputIt>> cursor(iterator_source<InputIt>(first, last), options, default_json_parsing());
        jsoncons::json_decoder<basic_json<char_type>> decoder;
        std::error_code ec;
        T val = decode_traits<T,char_type>::decode(cursor, decoder, ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec, cursor.line(), cursor.column()));
        }
        return val;
    }

    // With leading allocator_set parameter

    template <typename T,typename Source,typename Allocator,typename TempAllocator >
    typename std::enable_if<extension_traits::is_basic_json<T>::value &&
                            extension_traits::is_sequence_of<Source,typename T::char_type>::value,T>::type
    decode_json(const allocator_set<Allocator,TempAllocator>& alloc_set,
        const Source& s,
        const basic_json_decode_options<typename Source::value_type>& options = basic_json_decode_options<typename Source::value_type>())
    {
        using char_type = typename Source::value_type;

        json_decoder<T,TempAllocator> decoder(alloc_set.get_allocator(), alloc_set.get_temp_allocator());

        basic_json_reader<char_type, string_source<char_type>,TempAllocator> reader(s, decoder, options, alloc_set.get_temp_allocator());
        reader.read();
        if (!decoder.is_valid())
        {
            JSONCONS_THROW(ser_error(conv_errc::conversion_failed, reader.line(), reader.column()));
        }
        return decoder.get_result();
    }

    template <typename T,typename Source,typename Allocator,typename TempAllocator >
    typename std::enable_if<!extension_traits::is_basic_json<T>::value &&
                            extension_traits::is_char_sequence<Source>::value,T>::type
    decode_json(const allocator_set<Allocator,TempAllocator>& alloc_set,
        const Source& s,
        const basic_json_decode_options<typename Source::value_type>& options = basic_json_decode_options<typename Source::value_type>())
    {
        using char_type = typename Source::value_type;

        basic_json_cursor<char_type,string_source<char_type>,TempAllocator> cursor(s, options, default_json_parsing(), alloc_set.get_temp_allocator());
        json_decoder<basic_json<char_type,sorted_policy,TempAllocator>,TempAllocator> decoder(alloc_set.get_temp_allocator(), alloc_set.get_temp_allocator());

        std::error_code ec;
        T val = decode_traits<T,char_type>::decode(cursor, decoder, ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec, cursor.context().line(), cursor.context().column()));
        }
        return val;
    }

    template <typename T,typename CharT,typename Allocator,typename TempAllocator >
    typename std::enable_if<extension_traits::is_basic_json<T>::value,T>::type
    decode_json(const allocator_set<Allocator,TempAllocator>& alloc_set,
                std::basic_istream<CharT>& is,
                const basic_json_decode_options<CharT>& options = basic_json_decode_options<CharT>())
    {
        json_decoder<T,TempAllocator> decoder(alloc_set.get_allocator(), alloc_set.get_temp_allocator());

        basic_json_reader<CharT, stream_source<CharT>,TempAllocator> reader(is, decoder, options, alloc_set.get_temp_allocator());
        reader.read();
        if (!decoder.is_valid())
        {
            JSONCONS_THROW(ser_error(conv_errc::conversion_failed, reader.line(), reader.column()));
        }
        return decoder.get_result();
    }

    template <typename T,typename CharT,typename Allocator,typename TempAllocator >
    typename std::enable_if<!extension_traits::is_basic_json<T>::value,T>::type
    decode_json(const allocator_set<Allocator,TempAllocator>& alloc_set,
                std::basic_istream<CharT>& is,
                const basic_json_decode_options<CharT>& options = basic_json_decode_options<CharT>())
    {
        basic_json_cursor<CharT,stream_source<CharT>,TempAllocator> cursor(is, options, default_json_parsing(), alloc_set.get_temp_allocator());
        json_decoder<basic_json<CharT,sorted_policy,TempAllocator>,TempAllocator> decoder(alloc_set.get_temp_allocator(),alloc_set.get_temp_allocator());

        std::error_code ec;
        T val = decode_traits<T,CharT>::decode(cursor, decoder, ec);
        if (ec)
        {
            JSONCONS_THROW(ser_error(ec, cursor.context().line(), cursor.context().column()));
        }
        return val;
    }


} // namespace facebook::velox::jsoncons

#endif // JSONCONS_DECODE_JSON_HPP
