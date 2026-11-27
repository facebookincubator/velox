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

# The worker needs Python 3.10 and pyarrow; the tests also use pyspark to
# pickle functions and build ROW values.
find_package(Python3 3.10 COMPONENTS Interpreter)

if(Python3_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pyarrow, pyspark"
    RESULT_VARIABLE _python_worker_test_modules
    OUTPUT_QUIET
    ERROR_QUIET
  )
endif()

if(Python3_FOUND AND _python_worker_test_modules EQUAL 0)
  add_test(
    NAME velox_exec_python_worker_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v velox.exec.python.tests.test_worker
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
else()
  message(
    STATUS
    "Skipping velox_exec_python_worker_test: needs Python 3.10 or later with pyarrow and pyspark"
  )
endif()
