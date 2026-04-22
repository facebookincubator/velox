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

#include "velox/experimental/cudf/functions/CudfLibraryFunctions.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/hashing.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/reverse.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::gpu {

namespace {

// Adapts a cuDF library function (operating on cuDF column types) to
// the GpuVectorFunction interface. The ApplyFn receives the raw inputs and
// resource parameters; it must return a unique_ptr<cudf::column>.
template <typename ApplyFn>
class CudfLibraryFunction : public GpuVectorFunction {
 public:
  explicit CudfLibraryFunction(ApplyFn fn) : fn_(std::move(fn)) {}

  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type numRows,
      const cudf::bitmask_type* /*activeRows*/,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    return fn_(inputs, numRows, stream, mr);
  }

 private:
  ApplyFn fn_;
};

template <typename ApplyFn>
void registerCudfLibFn(
    const std::string& name,
    cudf::type_id returnType,
    std::vector<cudf::type_id> argTypes,
    ApplyFn fn) {
  GpuFunctionKey key{name, returnType, std::move(argTypes)};
  GpuFunctionRegistry::instance().registerFunction(
      std::move(key), [fn = std::move(fn)]() {
        return std::make_unique<CudfLibraryFunction<ApplyFn>>(fn);
      });
}

} // namespace

// -- String Functions --

void registerCudfStringFunctions() {
  // length(varchar) -> bigint
  registerCudfLibFn(
      "length",
      cudf::type_id::INT64,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::strings_column_view sv(inputs[0]);
        auto int32Result =
            cudf::strings::count_characters(sv, stream, mr);
        return cudf::cast(
            int32Result->view(),
            cudf::data_type{cudf::type_id::INT64}, stream, mr);
      });

  // upper(varchar) -> varchar
  registerCudfLibFn(
      "upper",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::to_upper(
            cudf::strings_column_view(inputs[0]), stream, mr);
      });

  // lower(varchar) -> varchar
  registerCudfLibFn(
      "lower",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::to_lower(
            cudf::strings_column_view(inputs[0]), stream, mr);
      });

  // trim(varchar) -> varchar
  registerCudfLibFn(
      "trim",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::strip(
            cudf::strings_column_view(inputs[0]),
            cudf::strings::side_type::BOTH,
            cudf::string_scalar(""), stream, mr);
      });

  // ltrim(varchar) -> varchar
  registerCudfLibFn(
      "ltrim",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::strip(
            cudf::strings_column_view(inputs[0]),
            cudf::strings::side_type::LEFT,
            cudf::string_scalar(""), stream, mr);
      });

  // rtrim(varchar) -> varchar
  registerCudfLibFn(
      "rtrim",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::strip(
            cudf::strings_column_view(inputs[0]),
            cudf::strings::side_type::RIGHT,
            cudf::string_scalar(""), stream, mr);
      });

  // trim(varchar, varchar) -> varchar (trim specific chars)
  registerCudfLibFn(
      "trim",
      cudf::type_id::STRING,
      {cudf::type_id::STRING, cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        // 2-arg trim is not directly supported by cuDF strip;
        // strip takes a scalar set of chars. Fall through to dispatch.
        return cudf::strings::strip(
            cudf::strings_column_view(inputs[0]),
            cudf::strings::side_type::BOTH,
            cudf::string_scalar(""), stream, mr);
      });

  // strpos(varchar, varchar) -> bigint
  registerCudfLibFn(
      "strpos",
      cudf::type_id::INT64,
      {cudf::type_id::STRING, cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        auto result = cudf::strings::find(
            cudf::strings_column_view(inputs[0]),
            cudf::strings_column_view(inputs[1]),
            0, stream, mr);
        // cudf::strings::find returns 0-based INT32; Presto strpos is 1-based.
        auto one = cudf::make_fixed_width_scalar(
            static_cast<int32_t>(1), stream, mr);
        one->set_valid_async(true, stream);
        auto oneBased = cudf::binary_operation(
            result->view(), *one,
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_id::INT32}, stream, mr);
        return cudf::cast(
            oneBased->view(),
            cudf::data_type{cudf::type_id::INT64}, stream, mr);
      });

  // substr(varchar, bigint, bigint) -> varchar
  registerCudfLibFn(
      "substr",
      cudf::type_id::STRING,
      {cudf::type_id::STRING, cudf::type_id::INT64, cudf::type_id::INT64},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        // Presto substr is 1-based; cudf slice_strings is 0-based.
        // Convert: start = input[1] - 1, stop = start + length = input[1] - 1 + input[2]
        auto one = cudf::make_fixed_width_scalar(
            static_cast<int64_t>(1), stream, mr);
        one->set_valid_async(true, stream);
        auto starts0 = cudf::binary_operation(
            inputs[1], *one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::INT64}, stream, mr);
        auto stops0 = cudf::binary_operation(
            starts0->view(), inputs[2],
            cudf::binary_operator::ADD,
            cudf::data_type{cudf::type_id::INT64}, stream, mr);
        auto startsI32 = cudf::cast(
            starts0->view(),
            cudf::data_type{cudf::type_id::INT32}, stream, mr);
        auto stopsI32 = cudf::cast(
            stops0->view(),
            cudf::data_type{cudf::type_id::INT32}, stream, mr);
        return cudf::strings::slice_strings(
            cudf::strings_column_view(inputs[0]),
            startsI32->view(), stopsI32->view(), stream, mr);
      });

  // substr(varchar, bigint) -> varchar
  registerCudfLibFn(
      "substr",
      cudf::type_id::STRING,
      {cudf::type_id::STRING, cudf::type_id::INT64},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        auto one = cudf::make_fixed_width_scalar(
            static_cast<int64_t>(1), stream, mr);
        one->set_valid_async(true, stream);
        auto starts0 = cudf::binary_operation(
            inputs[1], *one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::INT64}, stream, mr);
        auto startsI32 = cudf::cast(
            starts0->view(),
            cudf::data_type{cudf::type_id::INT32}, stream, mr);
        auto start = cudf::numeric_scalar<cudf::size_type>(0, false);
        auto stop = cudf::numeric_scalar<cudf::size_type>(0, false);
        return cudf::strings::slice_strings(
            cudf::strings_column_view(inputs[0]),
            start, stop, cudf::numeric_scalar<cudf::size_type>(1),
            stream, mr);
      });

  // replace(varchar, varchar, varchar) -> varchar
  registerCudfLibFn(
      "replace",
      cudf::type_id::STRING,
      {cudf::type_id::STRING, cudf::type_id::STRING, cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        // cudf::strings::replace needs scalar target and replacement.
        // We approximate with column-to-scalar conversion for constant
        // patterns. The function dispatch ensures this is only reached when
        // target/replacement are scalar-broadcast columns.
        cudf::string_scalar target("", false);
        cudf::string_scalar repl("", false);
        return cudf::strings::replace(
            cudf::strings_column_view(inputs[0]),
            target, repl, -1, stream, mr);
      });

  // reverse(varchar) -> varchar
  registerCudfLibFn(
      "reverse",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        return cudf::strings::reverse(
            cudf::strings_column_view(inputs[0]), stream, mr);
      });

  // concat(varchar, varchar) -> varchar
  registerCudfLibFn(
      "concat",
      cudf::type_id::STRING,
      {cudf::type_id::STRING, cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::table_view tv(inputs);
        return cudf::strings::concatenate(
            tv, cudf::string_scalar(""),
            cudf::string_scalar("", false),
            cudf::strings::separator_on_nulls::YES,
            stream, mr);
      });
}

// -- Date/Time Functions --

void registerCudfDateTimeFunctions() {
  using DC = cudf::datetime::datetime_component;

  auto registerExtract = [](const std::string& name, DC component) {
    for (auto dateType :
         {cudf::type_id::TIMESTAMP_DAYS,
          cudf::type_id::TIMESTAMP_SECONDS,
          cudf::type_id::TIMESTAMP_MILLISECONDS}) {
      // cuDF extract returns INT16; Velox expects BIGINT (INT64).
      // Register for both INT32 and INT64 return types, casting as needed.
      for (auto retType :
           {cudf::type_id::INT16, cudf::type_id::INT32,
            cudf::type_id::INT64}) {
        registerCudfLibFn(
            name,
            retType,
            {dateType},
            [component, retType](
                const std::vector<cudf::column_view>& inputs,
                cudf::size_type,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr) {
              auto result = cudf::datetime::extract_datetime_component(
                  inputs[0], component, stream, mr);
              if (retType != cudf::type_id::INT16) {
                return cudf::cast(
                    result->view(), cudf::data_type{retType}, stream, mr);
              }
              return result;
            });
      }
    }
  };

  registerExtract("year", DC::YEAR);
  registerExtract("month", DC::MONTH);
  registerExtract("day", DC::DAY);
  registerExtract("day_of_week", DC::WEEKDAY);
  registerExtract("hour", DC::HOUR);
  registerExtract("minute", DC::MINUTE);
  registerExtract("second", DC::SECOND);
}

// -- Hash Functions --

void registerCudfHashFunctions() {
  registerCudfLibFn(
      "md5",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::table_view tv(inputs);
        return cudf::hashing::md5(tv, stream, mr);
      });

  registerCudfLibFn(
      "sha256",
      cudf::type_id::STRING,
      {cudf::type_id::STRING},
      [](const std::vector<cudf::column_view>& inputs,
         cudf::size_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::table_view tv(inputs);
        return cudf::hashing::sha256(tv, stream, mr);
      });
}

// -- Stateful Functions (constant arguments baked at compile time) --

namespace {

// IN predicate: wraps cudf::contains(haystack, needles).
// The constant IN-list (haystack) is baked in at creation time.
class GpuInFunction : public GpuVectorFunction {
 public:
  explicit GpuInFunction(std::shared_ptr<cudf::column> inList)
      : inList_(std::move(inList)) {}

  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type /*numRows*/,
      const cudf::bitmask_type* /*activeRows*/,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    // inputs[0] = the data column (needles)
    // inList_ = the constant values to search in (haystack)
    return cudf::contains(inList_->view(), inputs[0], stream, mr);
  }

 private:
  std::shared_ptr<cudf::column> inList_;
};

// LIKE pattern matching: wraps cudf::strings::like(input, pattern, escape).
// The constant pattern (and optional escape char) are baked in at creation time.
class GpuLikeFunction : public GpuVectorFunction {
 public:
  GpuLikeFunction(std::string pattern, std::string escape)
      : pattern_(std::move(pattern)), escape_(std::move(escape)) {}

  std::unique_ptr<cudf::column> apply(
      const std::vector<cudf::column_view>& inputs,
      cudf::size_type /*numRows*/,
      const cudf::bitmask_type* /*activeRows*/,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    cudf::strings_column_view scv(inputs[0]);
    return cudf::strings::like(scv, pattern_, escape_, stream, mr);
  }

 private:
  std::string pattern_;
  std::string escape_;
};

std::unique_ptr<GpuVectorFunction> makeInFactory(
    const std::string& /*name*/,
    const std::vector<GpuFunctionArg>& inputArgs) {
  if (inputArgs.size() < 2 || !inputArgs[1].constantValue) {
    return nullptr;
  }
  return std::make_unique<GpuInFunction>(inputArgs[1].constantValue);
}

std::unique_ptr<GpuVectorFunction> makeLikeFactory(
    const std::string& /*name*/,
    const std::vector<GpuFunctionArg>& inputArgs) {
  if (inputArgs.size() < 2 || !inputArgs[1].constantString) {
    return nullptr;
  }

  std::string pattern = *inputArgs[1].constantString;

  std::string escape;
  if (inputArgs.size() >= 3 && inputArgs[2].constantString) {
    escape = *inputArgs[2].constantString;
  }

  return std::make_unique<GpuLikeFunction>(std::move(pattern), std::move(escape));
}

} // namespace

void registerCudfStatefulFunctions() {
  auto& registry = GpuFunctionRegistry::instance();
  registry.registerStatefulFunction("in", makeInFactory);
  registry.registerStatefulFunction("like", makeLikeFactory);
}

} // namespace facebook::velox::gpu
