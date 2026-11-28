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

# Embeds worker.py in the library as a raw string literal.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS worker.py)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/worker.py VELOX_PYTHON_WORKER_SCRIPT)
file(
  CONFIGURE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/WorkerScript.cpp
  CONTENT
    "#include \"velox/exec/python/WorkerScript.h\"
namespace facebook::velox::exec::python {
std::string_view workerScript() {
return R\"__WORKER__(@VELOX_PYTHON_WORKER_SCRIPT@)__WORKER__\";
}
}
"
  @ONLY
)

velox_add_library(
  velox_exec_python
  PythonEnvironmentResolver.cpp
  PythonWorker.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/WorkerScript.cpp
  HEADERS
  PythonEnvironmentResolver.h
  PythonWorker.h
  WorkerScript.h
)
velox_link_libraries(
  velox_exec_python
  velox_arrow_bridge
  velox_scoped_registry
  velox_vector
  arrow
  Folly::folly
  fmt::fmt
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
