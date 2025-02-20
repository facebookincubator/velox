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

#ifndef JSONCONS_EXT_JSONPATH_JSON_QUERY_HPP
#define JSONCONS_EXT_JSONPATH_JSON_QUERY_HPP

#include <type_traits>

#include "velox/external/jsoncons/allocator_set.hpp"
#include "velox/external/jsoncons/json_type_traits.hpp"
#include "velox/external/jsoncons/tag_type.hpp"
#include "velox/external/jsoncons/utility/extension_traits.hpp"

#include "velox/external/jsoncons/jsonpath/expression.hpp"
#include "velox/external/jsoncons/jsonpath/jsonpath_expression.hpp"
#include "velox/external/jsoncons/jsonpath/jsonpath_parser.hpp"
#include "velox/external/jsoncons/jsonpath/path_node.hpp"

namespace facebook::velox::jsoncons { 
namespace jsonpath {

    template <typename Json,typename JsonReference = const Json&>
    struct legacy_jsonpath_traits
    {
        using char_type = typename Json::char_type;
        using string_type = typename Json::string_type;
        using string_view_type = typename Json::string_view_type;
        using element_type = Json;
        using value_type = typename std::remove_cv<Json>::type;
        using reference = JsonReference;
        using const_reference = const value_type&;
        using pointer = typename std::conditional<std::is_const<typename std::remove_reference<reference>::type>::value,typename Json::const_pointer,typename Json::pointer>::type;
        using allocator_type = typename value_type::allocator_type;
        using evaluator_type = typename jsoncons::jsonpath::detail::jsonpath_evaluator<value_type, reference>;
        using path_node_type = basic_path_node<typename Json::char_type>;
        using path_expression_type = jsoncons::jsonpath::detail::path_expression<value_type,reference>;
        using path_pointer = const path_node_type*;
    };
     
    template <typename Json>
    Json json_query(const Json& root,
                    const typename Json::string_view_type& path, 
                    result_options options = result_options(),
                    const custom_functions<Json>& functions = custom_functions<Json>())
    {
        auto expr = make_expression<Json>(path, functions);
        return expr.evaluate(root, options);
    }

    template <typename Json,typename Callback>
    typename std::enable_if<extension_traits::is_binary_function_object<Callback,const typename Json::string_type&,const Json&>::value,void>::type
    json_query(const Json& root, 
               const typename Json::string_view_type& path, 
               Callback callback,
               result_options options = result_options(),
               const custom_functions<Json>& functions = custom_functions<Json>())
    {
        auto expr = make_expression<Json>(path, functions);
        expr.evaluate(root, callback, options);
    }

    template <typename Json,typename TempAllocator >
    Json json_query(const allocator_set<typename Json::allocator_type,TempAllocator>& alloc_set, 
        const Json& root, const typename Json::string_view_type& path, 
        result_options options = result_options(),
        const custom_functions<Json>& functions = custom_functions<Json>())
    {
        auto expr = make_expression<Json>(alloc_set, path, functions);
        return expr.evaluate(root, options);
    }

    template <typename Json,typename Callback,typename TempAllocator >
    typename std::enable_if<extension_traits::is_binary_function_object<Callback,const typename Json::string_type&,const Json&>::value,void>::type
    json_query(const allocator_set<typename Json::allocator_type,TempAllocator>& alloc_set, 
        const Json& root, const typename Json::string_view_type& path, 
        Callback callback,
        result_options options = result_options(),
        const custom_functions<Json>& functions = custom_functions<Json>())
    {
        auto expr = make_expression<Json>(alloc_set, path, functions);
        expr.evaluate(root, callback, options);
    }

    template <typename Json,typename T>
    typename std::enable_if<is_json_type_traits_specialized<Json,T>::value,void>::type
        json_replace(Json& root, const typename Json::string_view_type& path, T&& new_value,
                     const custom_functions<Json>& funcs = custom_functions<Json>())
    {
        using jsonpath_traits_type = jsoncons::jsonpath::legacy_jsonpath_traits<Json, Json&>;

        using value_type = typename jsonpath_traits_type::value_type;
        using reference = typename jsonpath_traits_type::reference;
        using evaluator_type = typename jsonpath_traits_type::evaluator_type;
        using path_expression_type = typename jsonpath_traits_type::path_expression_type;
        using path_node_type = typename jsonpath_traits_type::path_node_type;

        auto resources = jsoncons::make_unique<jsoncons::jsonpath::detail::static_resources<value_type>>(funcs);
        evaluator_type evaluator;
        path_expression_type expr = evaluator.compile(*resources, path);

        jsoncons::jsonpath::detail::eval_context<Json,reference> context;
        auto callback = [&new_value](const path_node_type&, reference v)
        {
            v = std::forward<T>(new_value);
        };

        result_options options = result_options::nodups | result_options::path | result_options::sort_descending;
        expr.evaluate(context, root, path_node_type{}, root, callback, options);
    }

    template <typename Json,typename T,typename TempAllocator >
    typename std::enable_if<is_json_type_traits_specialized<Json,T>::value,void>::type
        json_replace(const allocator_set<typename Json::allocator_type,TempAllocator>& alloc_set, 
            Json& root, const typename Json::string_view_type& path, T&& new_value,
            const custom_functions<Json>& funcs = custom_functions<Json>())
    {
        using jsonpath_traits_type = jsoncons::jsonpath::legacy_jsonpath_traits<Json, Json&>;

        using value_type = typename jsonpath_traits_type::value_type;
        using reference = typename jsonpath_traits_type::reference;
        using evaluator_type = typename jsonpath_traits_type::evaluator_type;
        using path_expression_type = typename jsonpath_traits_type::path_expression_type;
        using path_node_type = typename jsonpath_traits_type::path_node_type;

        auto resources = jsoncons::make_unique<jsoncons::jsonpath::detail::static_resources<value_type>>(funcs, alloc_set.get_allocator());
        evaluator_type evaluator{alloc_set.get_allocator()};
        path_expression_type expr = evaluator.compile(*resources, path);

        jsoncons::jsonpath::detail::eval_context<Json,reference> context{alloc_set.get_allocator()};
        auto callback = [&new_value](const path_node_type&, reference v)
        {
            v = Json(std::forward<T>(new_value), semantic_tag::none);
        };
        result_options options = result_options::nodups | result_options::path | result_options::sort_descending;
        expr.evaluate(context, root, path_node_type{}, root, callback, options);
    }

    template <typename Json,typename BinaryCallback>
    typename std::enable_if<extension_traits::is_binary_function_object<BinaryCallback,const typename Json::string_type&,Json&>::value,void>::type
    json_replace(Json& root, const typename Json::string_view_type& path , BinaryCallback callback, 
                 const custom_functions<Json>& funcs = custom_functions<Json>())
    {
        using jsonpath_traits_type = jsoncons::jsonpath::legacy_jsonpath_traits<Json, Json&>;

        using value_type = typename jsonpath_traits_type::value_type;
        using reference = typename jsonpath_traits_type::reference;
        using evaluator_type = typename jsonpath_traits_type::evaluator_type;
        using path_expression_type = typename jsonpath_traits_type::path_expression_type;
        using path_node_type = typename jsonpath_traits_type::path_node_type;

        auto resources = jsoncons::make_unique<jsoncons::jsonpath::detail::static_resources<value_type>>(funcs);
        evaluator_type evaluator;
        path_expression_type expr = evaluator.compile(*resources, path);

        jsoncons::jsonpath::detail::eval_context<Json,reference> context;

        auto f = [&callback](const path_node_type& path, reference val)
        {
            callback(to_basic_string(path), val);
        };
        result_options options = result_options::nodups | result_options::path | result_options::sort_descending;
        expr.evaluate(context, root, path_node_type{}, root, f, options);
    }

    template <typename Json,typename BinaryCallback,typename TempAllocator >
    typename std::enable_if<extension_traits::is_binary_function_object<BinaryCallback,const typename Json::string_type&,Json&>::value,void>::type
    json_replace(const allocator_set<typename Json::allocator_type,TempAllocator>& alloc_set, 
        Json& root, const typename Json::string_view_type& path , BinaryCallback callback, 
        const custom_functions<Json>& funcs = custom_functions<Json>())
    {
        using jsonpath_traits_type = jsoncons::jsonpath::legacy_jsonpath_traits<Json, Json&>;

        using value_type = typename jsonpath_traits_type::value_type;
        using reference = typename jsonpath_traits_type::reference;
        using evaluator_type = typename jsonpath_traits_type::evaluator_type;
        using path_expression_type = typename jsonpath_traits_type::path_expression_type;
        using path_node_type = typename jsonpath_traits_type::path_node_type;

        auto resources = jsoncons::make_unique<jsoncons::jsonpath::detail::static_resources<value_type>>(funcs, alloc_set.get_allocator());
        evaluator_type evaluator{alloc_set.get_allocator()};
        path_expression_type expr = evaluator.compile(*resources, path);

        jsoncons::jsonpath::detail::eval_context<Json,reference> context{alloc_set.get_allocator()};

        auto f = [&callback](const path_node_type& path, reference val)
        {
            callback(to_basic_string(path), val);
        };
        result_options options = result_options::nodups | result_options::path | result_options::sort_descending;
        expr.evaluate(context, root, path_node_type{}, root, f, options);
    }

    // Legacy replace function
    template <typename Json,typename UnaryCallback>
    typename std::enable_if<extension_traits::is_unary_function_object<UnaryCallback,Json>::value,void>::type
    json_replace(Json& root, const typename Json::string_view_type& path , UnaryCallback callback)
    {
        using jsonpath_traits_type = jsoncons::jsonpath::legacy_jsonpath_traits<Json, Json&>;

        using value_type = typename jsonpath_traits_type::value_type;
        using reference = typename jsonpath_traits_type::reference;
        using evaluator_type = typename jsonpath_traits_type::evaluator_type;
        using path_expression_type = typename jsonpath_traits_type::path_expression_type;
        using path_node_type = typename jsonpath_traits_type::path_node_type;

        auto resources = jsoncons::make_unique<jsoncons::jsonpath::detail::static_resources<value_type>>();
        evaluator_type evaluator;
        path_expression_type expr = evaluator.compile(*resources, path);

        jsoncons::jsonpath::detail::eval_context<Json,reference> context;
        auto f = [callback](const path_node_type&, reference v)
        {
            v = callback(v);
        };
        result_options options = result_options::nodups | result_options::path | result_options::sort_descending;
        expr.evaluate(context, root, path_node_type{}, root, f, options);
    }

} // namespace jsonpath
} // namespace facebook::velox::jsoncons

#endif // JSONCONS_EXT_JSONPATH_JSON_QUERY_HPP
