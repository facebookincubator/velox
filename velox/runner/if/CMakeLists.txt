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

include(FBThriftCppLibrary)

add_fbthrift_cpp_library(
  velox_local_runner_service_thrift
  LocalRunnerService.thrift
  SERVICES
  LocalRunnerService
)

target_compile_options(velox_local_runner_service_thrift PRIVATE -Wno-deprecated-declarations)
