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

#include <nvtx3/nvtx3.hpp>

namespace facebook::velox::cudf_velox {

// class NvtxHelper {
//  public:
//   NvtxHelper();

//  private:
//   nvtx3::color color_;
// };

/**
 * @brief Tag type for libkvikio's NVTX domain.
 */
struct velox_domain {
  static constexpr char const* name{"velox"};
};

using nvtx_registered_string_t = nvtx3::registered_string_in<velox_domain>;

#define VELOX_NVTX_FUNC_RANGE_IN_IMPL()                                        \
  static nvtx_registered_string_t const nvtx3_func_name__{                     \
      __PRETTY_FUNCTION__};                                                    \
  static ::nvtx3::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
  ::nvtx3::scoped_range_in<velox_domain> const nvtx3_range__{nvtx3_func_attr__};

#define VELOX_NVTX_OPERATOR_FUNC_RANGE() VELOX_NVTX_FUNC_RANGE_IN_IMPL()

} // namespace facebook::velox::cudf_velox
