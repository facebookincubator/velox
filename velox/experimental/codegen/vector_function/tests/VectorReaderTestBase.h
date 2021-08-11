/*
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

#include <gtest/gtest.h>
#include <memory>
#include "velox/experimental/codegen/vector_function/ComplexVectorReader.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::codegen {

class VectorReaderTestBase : public ::testing::Test {
 protected:
  std::unique_ptr<memory::ScopedMemoryPool> pool_ =
      memory::getDefaultScopedMemoryPool();

  testing::AssertionResult gtestMemcmp(void* lhs, void* rhs, size_t count) {
    return std::memcmp(lhs, rhs, count) ? testing::AssertionFailure()
                                        : testing::AssertionSuccess();
  }
};

class ComplexVectorReaderTest : public VectorReaderTestBase {
 protected:
  template <typename T>
  VectorPtr makeFlatVectorPtr(
      size_t flatVectorSize,
      const TypePtr type,
      memory::MemoryPool* pool) {
    auto vector = BaseVector::create(type, flatVectorSize, pool);
    return vector;
  }

  VectorPtr makeArrayVectorPtr(
      size_t arrayVectorSize,
      memory::MemoryPool* pool,
      const TypePtr type,
      VectorPtr elements) {
    BufferPtr offsets = AlignedBuffer::allocate<int32_t>(arrayVectorSize, pool);
    auto* offsetsPtr = offsets->asMutable<int32_t>();
    BufferPtr lengths =
        AlignedBuffer::allocate<vector_size_t>(arrayVectorSize, pool);
    auto* lengthsPtr = lengths->asMutable<vector_size_t>();
    BufferPtr nulls =
        AlignedBuffer::allocate<char>(bits::nbytes(arrayVectorSize), pool);
    auto* nullsPtr = nulls->asMutable<uint64_t>();

    size_t nullCount = 0;

    return std::make_shared<ArrayVector>(
        pool,
        type,
        nulls,
        arrayVectorSize,
        offsets,
        lengths,
        elements,
        nullCount);
  }
};

} // namespace facebook::velox::codegen
