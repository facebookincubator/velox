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
set(TEST_LINK_LIBS velox_vector_test_lib velox_temp_path GTest::gtest)

add_executable(velox_avro_reader_test AvroReaderTest.cpp)

add_test(
  NAME velox_avro_reader_test
  COMMAND velox_avro_reader_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(velox_avro_reader_test velox_dwio_avro_reader_register ${TEST_LINK_LIBS})
