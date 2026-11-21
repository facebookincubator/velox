# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Host half of the time zone conversions, compiled against real Velox: the
# device table of a zone is built from the tz::TimeZone the CPU resolves, and
# the session zone is read from QueryConfig. A shadow translation unit reaches
# it only through the POD-only GpuTimeZone.h and GpuTimeZoneDatabase.h.
add_library(velox_cudf_gpu_sfi_host GpuTimeZone.cpp GpuTimeZoneDatabase.cpp)

target_link_libraries(
  velox_cudf_gpu_sfi_host
  PRIVATE cudf::cudf velox_core velox_exception velox_functions_lib velox_type_tz
)
