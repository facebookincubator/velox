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

# The tests need Python 3.10 or later with pyarrow, which the worker imports,
# and pyspark, which the worker tests use to pickle functions and build ROW
# values.
find_package(Python3 3.10 COMPONENTS Interpreter)
if(Python3_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pyarrow, pyspark"
    RESULT_VARIABLE _python_test_modules
    OUTPUT_QUIET
    ERROR_QUIET
  )
endif()

if(NOT Python3_FOUND OR NOT _python_test_modules EQUAL 0)
  message(STATUS "Skipping the Python tests: they need Python 3.10 or later with pyarrow and pyspark")
  return()
endif()

add_test(
  NAME velox_exec_python_worker_test
  COMMAND "${Python3_EXECUTABLE}" -m unittest -v velox.exec.python.tests.test_worker
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(velox_exec_python_call_test PythonCallTest.cpp)
add_test(NAME velox_exec_python_call_test COMMAND velox_exec_python_call_test)
set_tests_properties(
  velox_exec_python_call_test
  PROPERTIES ENVIRONMENT "PYTHON_INTERPRETER=${Python3_EXECUTABLE}"
)
target_link_libraries(
  velox_exec_python_call_test
  velox_exec
  velox_exec_test_lib
  GTest::gtest
  GTest::gtest_main
)
