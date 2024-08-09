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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <functions/sparksql/Register.h>
#include <gtest/gtest.h>
#include <type/StringView.h>
#include <vector/ComplexVector.h>
#include <vector/FlatVector.h>
#include <vector/VectorSaver.h>
#include "velox/buffer/Buffer.h"

#include <iostream>

#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {
std::shared_ptr<memory::MemoryPool> pool_;
class VectorSaverBenchmark {
 public:
  VectorPtr generateTestVector(vector_size_t size, bool withNulls) {
    VectorPtr base = BaseVector::create(VARCHAR(), size, pool_.get());
    auto flat = std::dynamic_pointer_cast<FlatVector<StringView>>(base);
    for (int32_t i = 0; i < flat->size(); ++i) {
      if (withNulls && i % 3 == 0) {
        flat->setNull(i, true);
      } else {
        flat->set(i, testValue(base, i));
      }
    }
    return base;
  }
  static StringView testValue(VectorPtr vector, int32_t n) {
    std::stringstream out;
    out << n;
    for (int32_t i = 0; i < n % 20; ++i) {
      out << " " << i * i;
    }
    std::string str = out.str();
    if (str.size() > 12) {
      BufferPtr buffer = AlignedBuffer::allocate<char>(str.size(), pool_.get());
      std::dynamic_pointer_cast<FlatVector<StringView>>(vector)
          ->addStringBuffer(buffer);
      EXPECT_LE(str.size(), buffer->capacity());
      memcpy(buffer->asMutable<char>(), str.data(), str.size());
      return StringView(buffer->as<char>(), str.size());
    }
    return StringView(str.data(), str.size());
  }

  std::string runSaveVector(const VectorPtr& vector) {
    std::ostringstream out;
    saveVector(*vector, out);
    return out.str();
  }

  void runRestoreVector(const std::string& data) {
    std::istringstream dataStream(data);
    restoreVector(dataStream, pool_.get());
  }
};

VectorPtr data10k;
VectorPtr data100k;
VectorPtr data1000k;

std::string data10kString;
std::string data100kString;
std::string data1000kString;

std::shared_ptr<VectorSaverBenchmark> cb;

BENCHMARK(save_vector_10k) {
  cb->runSaveVector(data10k);
}

BENCHMARK(save_vector_100k) {
  cb->runSaveVector(data100k);
}
BENCHMARK(save_vector_1000k) {
  cb->runSaveVector(data1000k);
}

BENCHMARK(restore_vector_10k) {
  cb->runRestoreVector(data10kString);
}
BENCHMARK(restore_vector_100k) {
  cb->runRestoreVector(data100kString);
}

BENCHMARK(restore_vector_10000k) {
  cb->runRestoreVector(data1000kString);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize({});
  pool_ = {memory::memoryManager()->addLeafPool()};
  cb = std::make_shared<VectorSaverBenchmark>();
  data10k = cb->generateTestVector(10 * 1000, true);
  data100k = cb->generateTestVector(100 * 1000, true);
  data1000k = cb->generateTestVector(1000 * 1000, true);
  data10kString = cb->runSaveVector(data10k);
  data100kString = cb->runSaveVector(data100k);
  data1000kString = cb->runSaveVector(data1000k);
  folly::runBenchmarks();
  data10k.reset();
  data100k.reset();
  data1000k.reset();
  return 0;
}
