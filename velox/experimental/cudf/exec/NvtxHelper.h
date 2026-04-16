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

#include "velox/experimental/cudf/CudfConfig.h"

#include <nvtx3/nvtx3.hpp>

#include <glog/logging.h>

#include <optional>

namespace facebook::velox::cudf_velox {

class NvtxHelper {
 public:
  NvtxHelper();
  NvtxHelper(
      std::optional<nvtx3::color> color = std::nullopt,
      std::optional<int64_t> payload = std::nullopt,
      std::optional<std::string> extraInfo = std::nullopt)
      : color_(color.value_or(nvtx3::rgb{160, 82, 45} /* Sienna */)),
        payload_(payload),
        extraInfo_(extraInfo) {}

  nvtx3::color color_{nvtx3::rgb{125, 125, 125}}; // Gray
  std::optional<int64_t> payload_{};
  std::optional<std::string> extraInfo_{};
};

/**
 * @brief Tag type for Velox's NVTX domain.
 */
struct VeloxDomain {
  static constexpr char const* name{"velox"};
};

using NvtxRegisteredStringT = nvtx3::registered_string_in<VeloxDomain>;

/**
 * @brief Extracts class and function name from a pretty function string.
 *
 * This function parses a string like:
 * "virtual facebook::velox::RowVectorPtr
 * facebook::velox::cudf_velox::CudfHashAggregation::getOutput()" and returns
 * "CudfHashAggregation::getOutput"
 *
 * @param prettyFunction The string from __PRETTY_FUNCTION__
 * @return A simplified string in the format "classname::function"
 */
constexpr std::string_view extractClassAndFunction(
    std::string_view prettyFunction) {
  // Find the last occurrence of "::" before the opening parenthesis
  auto parenPos = prettyFunction.find('(');
  if (parenPos == std::string_view::npos) {
    parenPos = prettyFunction.size();
  }

  auto lastColonPos = prettyFunction.rfind("::", parenPos);
  if (lastColonPos == std::string_view::npos) {
    return prettyFunction.substr(0, parenPos); // No class name found
  }

  // Find the previous "::" to get the start of the class name
  auto prevColonPos = prettyFunction.rfind("::", lastColonPos - 1);
  if (prevColonPos == std::string_view::npos) {
    return prettyFunction.substr(0, parenPos); // No namespace found
  }

  // Return the class and function name
  return prettyFunction.substr(prevColonPos + 2, parenPos - prevColonPos - 2);
}

/// Extracts only the function name (without class) from __PRETTY_FUNCTION__.
/// For example, "CudfOperatorBase::addInput" becomes "addInput".
constexpr std::string_view extractFunctionName(
    std::string_view prettyFunction) {
  auto parenPos = prettyFunction.find('(');
  if (parenPos == std::string_view::npos) {
    parenPos = prettyFunction.size();
  }

  auto lastColonPos = prettyFunction.rfind("::", parenPos);
  if (lastColonPos == std::string_view::npos) {
    return prettyFunction.substr(0, parenPos);
  }

  return prettyFunction.substr(lastColonPos + 2, parenPos - lastColonPos - 2);
}

#define VELOX_NVTX_OPERATOR_FUNC_RANGE(...)                                       \
  static_assert(                                                                  \
      std::is_base_of<NvtxHelper, std::remove_pointer<decltype(this)>::type>::    \
          value,                                                                  \
      "VELOX_NVTX_OPERATOR_FUNC_RANGE can only be used"                           \
      " in Operators derived from NvtxHelper");                                   \
  static std::string const nvtx3_func_name__{                                     \
      std::string(extractClassAndFunction(__PRETTY_FUNCTION__))};                 \
  std::string const nvtx3_func_display_name__{__VA_OPT__(                         \
      std::string(__VA_ARGS__) +                                                  \
      "::" + std::string(extractFunctionName(__PRETTY_FUNCTION__)))               \
                                                  __VA_OPT__(+) ""};              \
  std::string const nvtx3_func_extra_info__{                                      \
      (nvtx3_func_display_name__.empty() ? nvtx3_func_name__                      \
                                         : nvtx3_func_display_name__) +           \
      " " + this->extraInfo_.value_or("")};                                       \
  ::nvtx3::event_attributes const nvtx3_func_attr__{                            \
      this->payload_.has_value() ?                                               \
          ::nvtx3::event_attributes{nvtx3_func_extra_info__, this->color_,      \
                                   nvtx3::payload{this->payload_.value()}} :    \
          ::nvtx3::event_attributes{nvtx3_func_extra_info__, this->color_}}; \
  ::nvtx3::scoped_range_in<VeloxDomain> const nvtx3_range__{nvtx3_func_attr__};

#define VELOX_NVTX_OPERATOR_FUNC_RANGE_IF(condition, ...)                          \
  static_assert(                                                                   \
      std::is_base_of<NvtxHelper, std::remove_pointer<decltype(this)>::type>::     \
          value,                                                                   \
      "VELOX_NVTX_OPERATOR_FUNC_RANGE_IF can only be used"                         \
      " in Operators derived from NvtxHelper");                                    \
  ::nvtx3::detail::optional_scoped_range_in<VeloxDomain> nvtx3_opt_range__;        \
  if (condition) {                                                                 \
    static std::string const nvtx3_func_name__{                                    \
        std::string(extractClassAndFunction(__PRETTY_FUNCTION__))};                \
    std::string const nvtx3_func_display_name__{__VA_OPT__(                        \
        std::string(__VA_ARGS__) +                                                 \
        "::" + std::string(extractFunctionName(__PRETTY_FUNCTION__)))              \
                                                    __VA_OPT__(+) ""};             \
    std::string const nvtx3_func_extra_info__{                                     \
        (nvtx3_func_display_name__.empty() ? nvtx3_func_name__                     \
                                           : nvtx3_func_display_name__) +          \
        " " + this->extraInfo_.value_or("")};                                      \
    if (CudfConfig::getInstance().debugEnabled) {                                  \
      VLOG(2) << "Calling " << nvtx3_func_extra_info__;                            \
    }                                                                              \
    ::nvtx3::event_attributes const nvtx3_func_attr__{                           \
        this->payload_.has_value() ?                                             \
            ::nvtx3::event_attributes{nvtx3_func_extra_info__, this->color_,    \
                                     nvtx3::payload{this->payload_.value()}} :  \
            ::nvtx3::event_attributes{nvtx3_func_extra_info__, this->color_}}; \
    nvtx3_opt_range__.begin(nvtx3_func_attr__);                                    \
  }

#define VELOX_NVTX_PRETTY_FUNC_RANGE()                                         \
  static NvtxRegisteredStringT const nvtx3_func_name__{                        \
      std::string(__func__) + " " + std::string(__PRETTY_FUNCTION__)};         \
  static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
  ::nvtx3::scoped_range_in<VeloxDomain> const nvtx3_range__{nvtx3_func_attr__};

#define VELOX_NVTX_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(VeloxDomain)

} // namespace facebook::velox::cudf_velox
