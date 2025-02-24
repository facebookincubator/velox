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

#ifndef JSONCONS_STAJ_EVENT_READER_HPP
#define JSONCONS_STAJ_EVENT_READER_HPP

#include <array> // std::array
#include <cstddef>
#include <cstdint>
#include <functional> // std::function
#include <ios>
#include <memory> // std::allocator
#include <system_error>

#include "velox/external/jsoncons/conv_error.hpp"
#include "velox/external/jsoncons/detail/write_number.hpp"
#include "velox/external/jsoncons/item_event_visitor.hpp"
#include "velox/external/jsoncons/json_parser.hpp"
#include "velox/external/jsoncons/json_type_traits.hpp"
#include "velox/external/jsoncons/ser_context.hpp"
#include "velox/external/jsoncons/sink.hpp"
#include "velox/external/jsoncons/staj_event.hpp"
#include "velox/external/jsoncons/tag_type.hpp"
#include "velox/external/jsoncons/typed_array_view.hpp"
#include "velox/external/jsoncons/utility/bigint.hpp"
#include "velox/external/jsoncons/value_converter.hpp"

namespace facebook::velox::jsoncons {

    // basic_item_event_receiver

    enum class item_event_reader_state
    {
        typed_array = 1,
        multi_dim,
        shape
    };

    template <typename CharT>
    class basic_item_event_receiver : public basic_item_event_visitor<CharT>
    {
        using super_type = basic_item_event_visitor<CharT>;
    public:
        using char_type = CharT;
        using typename super_type::string_view_type;
    private:
        std::function<bool(const basic_staj_event<CharT>&, const ser_context&)> pred_;
        basic_staj_event<CharT> event_;

        item_event_reader_state state_;
        typed_array_view data_;
        jsoncons::span<const size_t> shape_;
        std::size_t index_;
    public:
        basic_item_event_receiver()
            : pred_(accept), event_(staj_event_type::null_value),
              state_(), data_(), shape_(), index_(0)
        {
        }

        basic_item_event_receiver(std::function<bool(const basic_staj_event<CharT>&, const ser_context&)> pred)
            : pred_(pred), event_(staj_event_type::null_value),
              state_(), data_(), shape_(), index_(0)
        {
        }

        void reset()
        {
            event_ = staj_event_type::null_value;
            state_ = {};
            data_ = {};
            shape_ = {};
            index_ = 0;
        }

        const basic_staj_event<CharT>& event() const
        {
            return event_;
        }

        bool in_available() const
        {
            return state_ != item_event_reader_state();
        }

        void send_available(std::error_code& ec)
        {
            switch (state_)
            {
                case item_event_reader_state::typed_array:
                    advance_typed_array(ec);
                    break;
                case item_event_reader_state::multi_dim:
                case item_event_reader_state::shape:
                    advance_multi_dim(ec);
                    break;
                default:
                    break;
            }
        }

        bool is_typed_array() const
        {
            return data_.type() != typed_array_type();
        }

        item_event_reader_state state() const
        {
            return state_;
        }

        void advance_typed_array(std::error_code& ec)
        {
            if (is_typed_array())
            {
                if (index_ < data_.size())
                {
                    switch (data_.type())
                    {
                        case typed_array_type::uint8_value:
                        {
                            this->uint64_value(data_.data(uint8_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::uint16_value:
                        {
                            this->uint64_value(data_.data(uint16_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::uint32_value:
                        {
                            this->uint64_value(data_.data(uint32_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::uint64_value:
                        {
                            this->uint64_value(data_.data(uint64_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::int8_value:
                        {
                            this->int64_value(data_.data(int8_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::int16_value:
                        {
                            this->int64_value(data_.data(int16_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::int32_value:
                        {
                            this->int64_value(data_.data(int32_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::int64_value:
                        {
                            this->int64_value(data_.data(int64_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::half_value:
                        {
                            this->half_value(data_.data(half_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::float_value:
                        {
                            this->double_value(data_.data(float_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        case typed_array_type::double_value:
                        {
                            this->double_value(data_.data(double_array_arg)[index_], semantic_tag::none, ser_context(), ec);
                            break;
                        }
                        default:
                            break;
                    }
                    ++index_;
                }
                else
                {
                    this->end_array();
                    state_ = item_event_reader_state();
                    data_ = typed_array_view();
                    index_ = 0;
                }
            }
        }

        void advance_multi_dim(std::error_code& ec)
        {
            if (shape_.size() != 0)
            {
                if (state_ == item_event_reader_state::multi_dim)
                {
                    this->begin_array(shape_.size(), semantic_tag::none, ser_context(), ec);
                    state_ = item_event_reader_state::shape;
                }
                else if (index_ < shape_.size())
                {
                    this->uint64_value(shape_[index_], semantic_tag::none, ser_context(), ec);
                    ++index_;
                }
                else
                {
                    state_ = item_event_reader_state();
                    this->end_array(ser_context(), ec);
                    shape_ = jsoncons::span<const size_t>();
                    index_ = 0;
                }
            }
        }

        bool dump(basic_item_event_visitor<CharT>& visitor, const ser_context& context, std::error_code& ec)
        {
            bool more = true;
            if (is_typed_array())
            {
                if (index_ != 0)
                {
                    more = event().send_value_event(visitor, context, ec);
                    while (more && is_typed_array())
                    {
                        if (index_ < data_.size())
                        {
                            switch (data_.type())
                            {
                                case typed_array_type::uint8_value:
                                {
                                    more = visitor.uint64_value(data_.data(uint8_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::uint16_value:
                                {
                                    more = visitor.uint64_value(data_.data(uint16_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::uint32_value:
                                {
                                    more = visitor.uint64_value(data_.data(uint32_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::uint64_value:
                                {
                                    more = visitor.uint64_value(data_.data(uint64_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::int8_value:
                                {
                                    more = visitor.int64_value(data_.data(int8_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::int16_value:
                                {
                                    more = visitor.int64_value(data_.data(int16_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::int32_value:
                                {
                                    more = visitor.int64_value(data_.data(int32_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::int64_value:
                                {
                                    more = visitor.int64_value(data_.data(int64_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::float_value:
                                {
                                    more = visitor.double_value(data_.data(float_array_arg)[index_]);
                                    break;
                                }
                                case typed_array_type::double_value:
                                {
                                    more = visitor.double_value(data_.data(double_array_arg)[index_]);
                                    break;
                                }
                                default:
                                    break;
                            }
                            ++index_;
                        }
                        else
                        {
                            more = visitor.end_array();
                            state_ = item_event_reader_state();
                            data_ = typed_array_view();
                            index_ = 0;
                        }
                    }
                }
                else
                {
                    switch (data_.type())
                    {
                        case typed_array_type::uint8_value:
                        {
                            more = visitor.typed_array(data_.data(uint8_array_arg));
                            break;
                        }
                        case typed_array_type::uint16_value:
                        {
                            more = visitor.typed_array(data_.data(uint16_array_arg));
                            break;
                        }
                        case typed_array_type::uint32_value:
                        {
                            more = visitor.typed_array(data_.data(uint32_array_arg));
                            break;
                        }
                        case typed_array_type::uint64_value:
                        {
                            more = visitor.typed_array(data_.data(uint64_array_arg));
                            break;
                        }
                        case typed_array_type::int8_value:
                        {
                            more = visitor.typed_array(data_.data(int8_array_arg));
                            break;
                        }
                        case typed_array_type::int16_value:
                        {
                            more = visitor.typed_array(data_.data(int16_array_arg));
                            break;
                        }
                        case typed_array_type::int32_value:
                        {
                            more = visitor.typed_array(data_.data(int32_array_arg));
                            break;
                        }
                        case typed_array_type::int64_value:
                        {
                            more = visitor.typed_array(data_.data(int64_array_arg));
                            break;
                        }
                        case typed_array_type::float_value:
                        {
                            more = visitor.typed_array(data_.data(float_array_arg));
                            break;
                        }
                        case typed_array_type::double_value:
                        {
                            more = visitor.typed_array(data_.data(double_array_arg));
                            break;
                        }
                        default:
                            break;
                    }

                    state_ = item_event_reader_state();
                    data_ = typed_array_view();
                }
            }
            else
            {
                more = event().send_value_event(visitor, context, ec);
            }
            return more;
        }

    private:
        static constexpr bool accept(const basic_staj_event<CharT>&, const ser_context&) 
        {
            return true;
        }

        bool visit_begin_object(semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::begin_object, tag);
            return !pred_(event_, context);
        }

        bool visit_begin_object(std::size_t length, semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::begin_object, length, tag);
            return !pred_(event_, context);
        }

        bool visit_end_object(const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::end_object);
            return !pred_(event_, context);
        }

        bool visit_begin_array(semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::begin_array, tag);
            return !pred_(event_, context);
        }

        bool visit_begin_array(std::size_t length, semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::begin_array, length, tag);
            return !pred_(event_, context);
        }

        bool visit_end_array(const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::end_array);
            return !pred_(event_, context);
        }

        bool visit_null(semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(staj_event_type::null_value, tag);
            return !pred_(event_, context);
        }

        bool visit_bool(bool value, semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(value, tag);
            return !pred_(event_, context);
        }

        bool visit_string(const string_view_type& s, semantic_tag tag, const ser_context& context, std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(s, staj_event_type::string_value, tag);
            return !pred_(event_, context);
        }

        bool visit_byte_string(const byte_string_view& s, 
                               semantic_tag tag,
                               const ser_context& context,
                               std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(s, staj_event_type::byte_string_value, tag);
            return !pred_(event_, context);
        }

        bool visit_byte_string(const byte_string_view& s, 
                               uint64_t ext_tag,
                               const ser_context& context,
                               std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(s, staj_event_type::byte_string_value, ext_tag);
            return !pred_(event_, context);
        }

        bool visit_uint64(uint64_t value, 
                             semantic_tag tag, 
                             const ser_context& context,
                             std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(value, tag);
            return !pred_(event_, context);
        }

        bool visit_int64(int64_t value, 
                      semantic_tag tag,
                      const ser_context& context,
                      std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(value, tag);
            return !pred_(event_, context);
        }

        bool visit_half(uint16_t value, 
                     semantic_tag tag,
                     const ser_context& context,
                     std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(half_arg, value, tag);
            return !pred_(event_, context);
        }

        bool visit_double(double value, 
                       semantic_tag tag, 
                       const ser_context& context,
                       std::error_code&) override
        {
            event_ = basic_staj_event<CharT>(value, tag);
            return !pred_(event_, context);
        }

        bool visit_typed_array(const jsoncons::span<const uint8_t>& v, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(v.data(), v.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const uint16_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const uint32_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const uint64_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const int8_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const int16_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const int32_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const int64_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(half_arg_t, const jsoncons::span<const uint16_t>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const float>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }

        bool visit_typed_array(const jsoncons::span<const double>& data, 
                            semantic_tag tag,
                            const ser_context& context,
                            std::error_code& ec) override
        {
            state_ = item_event_reader_state::typed_array;
            data_ = typed_array_view(data.data(), data.size());
            index_ = 0;
            return this->begin_array(tag, context, ec);
        }
    /*
        bool visit_typed_array(const jsoncons::span<const float128_type>&, 
                            semantic_tag,
                            const ser_context&,
                            std::error_code&) override
        {
            return true;
        }
    */
        bool visit_begin_multi_dim(const jsoncons::span<const size_t>& shape,
                                semantic_tag tag,
                                const ser_context& context, 
                                std::error_code& ec) override
        {
            state_ = item_event_reader_state::multi_dim;
            shape_ = shape;
            return this->begin_array(2, tag, context, ec);
        }

        bool visit_end_multi_dim(const ser_context& context,
                              std::error_code& ec) override
        {
            return this->end_array(context, ec);
        }

        void visit_flush() override
        {
        }
    };

    // basic_staj_event_reader

    template <typename CharT>
    class basic_staj_event_reader
    {
    public:
        virtual ~basic_staj_event_reader() = default;

        virtual void array_expected(std::error_code& ec)
        {
            if (!(current().event_type() == staj_event_type::begin_array || current().event_type() == staj_event_type::byte_string_value))
            {
                ec = conv_errc::not_vector;
            }
        }

        virtual bool done() const = 0;

        virtual const basic_staj_event<CharT>& current() const = 0;

        virtual void read_to(basic_item_event_visitor<CharT>& visitor) = 0;

        virtual void read_to(basic_item_event_visitor<CharT>& visitor,
                             std::error_code& ec) = 0;

        virtual void next() = 0;

        virtual void next(std::error_code& ec) = 0;

        virtual const ser_context& context() const = 0;
    };

    template <typename CharT>
    class basic_staj2_filter_view : basic_staj_event_reader<CharT>
    {
        basic_staj_event_reader<CharT>* cursor_;
        std::function<bool(const basic_staj_event<CharT>&, const ser_context&)> pred_;
    public:
        basic_staj2_filter_view(basic_staj_event_reader<CharT>& cursor,
                         std::function<bool(const basic_staj_event<CharT>&, const ser_context&)> pred)
            : cursor_(std::addressof(cursor)), pred_(pred)
        {
            while (!done() && !pred_(current(),context()))
            {
                cursor_->next();
            }
        }

        bool done() const override
        {
            return cursor_->done();
        }

        const basic_staj_event<CharT>& current() const override
        {
            return cursor_->current();
        }

        void read_to(basic_item_event_visitor<CharT>& visitor) override
        {
            cursor_->read_to(visitor);
        }

        void read_to(basic_item_event_visitor<CharT>& visitor,
                     std::error_code& ec) override
        {
            cursor_->read_to(visitor, ec);
        }

        void next() override
        {
            cursor_->next();
            while (!done() && !pred_(current(),context()))
            {
                cursor_->next();
            }
        }

        void next(std::error_code& ec) override
        {
            cursor_->next(ec);
            while (!done() && !pred_(current(),context()) && !ec)
            {
                cursor_->next(ec);
            }
        }

        const ser_context& context() const override
        {
            return cursor_->context();
        }

        friend
            basic_staj2_filter_view<CharT> operator|(basic_staj2_filter_view& cursor,
                                          std::function<bool(const basic_staj_event<CharT>&, const ser_context&)> pred)
        {
            return basic_staj2_filter_view<CharT>(cursor, pred);
        }
    };

    using item_event = basic_staj_event<char>;
    using witem_event = basic_staj_event<wchar_t>;

    using staj_event_reader = basic_staj_event_reader<char>;
    using wstaj_event_reader = basic_staj_event_reader<wchar_t>;

    using staj2_filter_view = basic_staj2_filter_view<char>;
    using wstaj2_filter_view = basic_staj2_filter_view<wchar_t>;

} // namespace facebook::velox::jsoncons

#endif // JSONCONS_STAJ_EVENT_READER_HPP
