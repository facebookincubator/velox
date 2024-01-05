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

#include "velox/exec/SetAccumulator.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Values.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/VectorTestUtils.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorMakerStats.h"

using namespace facebook::velox;
using namespace facebook::velox::aggregate::prestosql;
using namespace facebook::velox::test;

namespace facebook::velox::exec::test {

class SetAccumulatorTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
  }

  template <TypeKind TKind>
  void testSpillRoundtrip() {
    const auto type{createScalarType(TKind)};
    using TNativeType = typename TypeTraits<TKind>::NativeType;

    SetAccumulator<TNativeType> source{type, allocator_.get()};

    auto generated =
        genTestData<TNativeType>(100, type, true /* includeNulls */);
    auto data = generated.data();
    auto flatVector = makeNullableFlatVector(data, type);

    DecodedVector decoded(*flatVector);

    for (auto i = 0; i < decoded.size(); ++i) {
      source.addValue(decoded, i, allocator_.get());
    }

    SetAccumulator<TNativeType> destination{type, allocator_.get()};

    {
      auto sourceVector = BaseVector::create(VARCHAR(), 100, pool_.get());
      auto* flatSource = sourceVector->asFlatVector<StringView>();
      flatSource->resize(1);

      {
        auto* rawBuffer = flatSource->getRawStringBufferWithSpace(
            source.maxSpillSize(), true);

        source.extractForSpill(rawBuffer, source.maxSpillSize());

        flatSource->setNoCopy(0, StringView(rawBuffer, source.maxSpillSize()));
      }

      destination.addFromSpill(*flatSource, allocator_.get());
    }

    {
      auto sourceVector = BaseVector::create(type, 100, pool_.get());
      auto* flatsource = sourceVector->asFlatVector<TNativeType>();
      flatsource->resize(source.size());

      source.extractValues(*flatsource, 0);

      auto destinationVector = BaseVector::create(type, 100, pool_.get());
      auto* flatDestination = destinationVector->asFlatVector<TNativeType>();
      flatDestination->resize(destination.size());

      destination.extractValues(*flatDestination, 0);

      static const CompareFlags kCompareFlags{
          true, // nullsFirst
          true, // ascending
          true, // equalsOnly
          CompareFlags::NullHandlingMode::kNullAsValue};

      EXPECT_TRUE(
          flatDestination->compare(flatsource, 0, 0, kCompareFlags)
              .value_or(-1) == 0);
    }
  }

  template <typename T>
  FlatVectorPtr<EvalType<T>> makeNullableFlatVector(
      const std::vector<std::optional<T>>& data,
      const TypePtr& type = CppToType<T>::create()) {
    return vectorMaker_.flatVectorNullable(data, type);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  std::unique_ptr<HashStringAllocator> allocator_{
      std::make_unique<HashStringAllocator>(pool())};
};

TEST_F(SetAccumulatorTest, SpillRoundTripBoolean) {
  testSpillRoundtrip<TypeKind::BOOLEAN>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripTinyInt) {
  testSpillRoundtrip<TypeKind::TINYINT>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripSmallInt) {
  testSpillRoundtrip<TypeKind::SMALLINT>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripInteger) {
  testSpillRoundtrip<TypeKind::INTEGER>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripBigInt) {
  testSpillRoundtrip<TypeKind::BIGINT>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripDouble) {
  testSpillRoundtrip<TypeKind::DOUBLE>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripVarchar) {
  testSpillRoundtrip<TypeKind::VARCHAR>();
}

TEST_F(SetAccumulatorTest, SpillRoundTripRow) {
  const auto type{ROW({{"c0", INTEGER()}})};

  SetAccumulator<ComplexType> source{type, allocator_.get()};

  auto baseVector = BaseVector::create(type, 1, pool_.get());
  auto rowVector = baseVector->asUnchecked<RowVector>();
  auto flatVector = rowVector->childAt(0)->asFlatVector<int32_t>();
  flatVector->resize(100);
  for (int i = 0; i < flatVector->size(); ++i) {
    flatVector->set(i, i);
  }

  DecodedVector decoded(*baseVector);

  for (auto i = 0; i < decoded.size(); ++i) {
    source.addValue(decoded, i, allocator_.get());
  }

  SetAccumulator<ComplexType> destination{type, allocator_.get()};

  {
    auto sourceVector = BaseVector::create(VARCHAR(), 100, pool_.get());
    auto* flatSource = sourceVector->asFlatVector<StringView>();
    flatSource->resize(1);

    {
      auto* rawBuffer =
          flatSource->getRawStringBufferWithSpace(source.maxSpillSize(), true);

      source.extractForSpill(rawBuffer, source.maxSpillSize());

      flatSource->setNoCopy(0, StringView(rawBuffer, source.maxSpillSize()));
    }

    destination.addFromSpill(*flatSource, allocator_.get());
  }

  {
    auto sourceVector = BaseVector::create(type, 1, pool_.get());
    auto* rowsource = sourceVector->asUnchecked<RowVector>();
    rowsource->resize(source.size());

    source.extractValues(*rowsource, 0);

    auto destinationVector = BaseVector::create(type, 1, pool_.get());
    auto* rowdestination = destinationVector->asUnchecked<RowVector>();
    rowdestination->resize(destination.size());

    destination.extractValues(*rowdestination, 0);

    static const CompareFlags kCompareFlags{
        true, // nullsFirst
        true, // ascending
        true, // equalsOnly
        CompareFlags::NullHandlingMode::kNullAsValue};

    EXPECT_TRUE(
        rowdestination->childAt(0)
            ->asFlatVector<int32_t>()
            ->compare(
                rowsource->childAt(0)->asFlatVector<int32_t>(),
                0,
                0,
                kCompareFlags)
            .value_or(-1) == 0);
  }
}

} // namespace facebook::velox::exec::test
