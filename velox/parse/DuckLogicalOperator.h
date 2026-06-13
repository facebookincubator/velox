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

// This header used to carry copies of DuckDB logical operator definitions to
// avoid including duckdb-internal.hpp. DuckDB now exposes these definitions in
// public operator headers, so include the specific logical nodes Velox
// inspects.

#include <duckdb/planner/operator/logical_aggregate.hpp> // @manual
#include <duckdb/planner/operator/logical_any_join.hpp> // @manual
#include <duckdb/planner/operator/logical_comparison_join.hpp> // @manual
#include <duckdb/planner/operator/logical_cross_product.hpp> // @manual
#include <duckdb/planner/operator/logical_delim_get.hpp> // @manual
#include <duckdb/planner/operator/logical_filter.hpp> // @manual
#include <duckdb/planner/operator/logical_get.hpp> // @manual
#include <duckdb/planner/operator/logical_limit.hpp> // @manual
#include <duckdb/planner/operator/logical_order.hpp> // @manual
#include <duckdb/planner/operator/logical_projection.hpp> // @manual
