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

#include "SortSpillInputBenchmarkBase.h"
#include "velox/serializers/PrestoSerializer.h"

#include <gflags/gflags.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;

bool containsString(TypePtr type) {
  if (type->kind() == TypeKind::VARCHAR ||
      type->kind() == TypeKind::VARBINARY) {
    return true;
  }
  for (auto i = 0; i < type->size(); ++i) {
    if (containsString(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

void runTest(RowTypePtr rowType, bool serializeRowContainer, int strLength) {
  auto test = std::make_unique<test::SortSpillInputBenchmarkBase>();
  test->setUp(rowType, serializeRowContainer, strLength);
  test->run();
  test->printStats();
  test->cleanup();
}

void doTest(RowTypePtr rowType) {
  LOG(INFO) << "=====================BENCHMARK===========================";
  bool conf[] = {false, true};
  int strLength[] = {10, 50, 100};
  LOG(INFO) << "row type: " << rowType->toString();
  for (auto serializeRowContainer : conf) {
    if (containsString(rowType)) {
      for (auto length : strLength) {
        LOG(INFO)
            << "-----------------------------------------------------------";
        LOG(INFO) << "serializeRowContainer: "
                  << (serializeRowContainer ? "true" : "false")
                  << " string max length: " << length;
        runTest(rowType, serializeRowContainer, length);
      }
    } else {
      LOG(INFO)
          << "-----------------------------------------------------------";
      LOG(INFO) << "serializeRowContainer: "
                << (serializeRowContainer ? "true" : "false");
      runTest(rowType, serializeRowContainer, 0);
    }
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  filesystems::registerLocalFileSystem();

  RowTypePtr rowTypes[] = {
      ROW({"c0", "c1", "c2"}, {INTEGER(), BIGINT(), INTEGER()}),
      ROW({"c0", "c1", "c2"}, {INTEGER(), BIGINT(), VARBINARY()}),
      ROW({"c0", "c1", "c2"}, {INTEGER(), BIGINT(), ARRAY(INTEGER())}),
      ROW({"c0", "c1", "c2"}, {INTEGER(), BIGINT(), ARRAY(VARBINARY())}),
      ROW({"c0", "c1", "c2"},
          {INTEGER(), BIGINT(), MAP(INTEGER(), VARBINARY())}),
      ROW({"c0", "c1", "c2"},
          {INTEGER(),
           BIGINT(),
           ROW({"int", "str"}, {INTEGER(), VARBINARY()})})};
  for (auto rowType : rowTypes) {
    doTest(rowType);
  }
  return 0;
}
