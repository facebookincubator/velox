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

#include "velox/dwio/common/tests/UnitLoaderBaseTest.h"

using facebook::velox::dwio::common::LoadUnit;
using facebook::velox::dwio::common::test::getUnitsLoadedWithFalse;
using facebook::velox::dwio::common::test::LoadUnitMock;
using facebook::velox::dwio::common::test::ReaderMock;

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<UnitLoaderFactoryType>::testNoUnitButSkip() {
  UnitLoaderFactoryType factory = createFactory();
  std::vector<std::unique_ptr<LoadUnit>> units;

  EXPECT_NO_THROW(factory.create(std::move(units), 0));

  std::vector<std::unique_ptr<LoadUnit>> units2;
  VELOX_ASSERT_THROW(
      factory.create(std::move(units2), 1),
      "Can only skip up to the past-the-end row of the file.");
}

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<UnitLoaderFactoryType>::testInitialSkip() {
  auto getFactoryWithSkip = [this](uint64_t skipToRow) {
    auto factory = createFactory();
    std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(3));
    std::vector<std::unique_ptr<LoadUnit>> units;
    units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));
    units.push_back(std::make_unique<LoadUnitMock>(20, 0, unitsLoaded, 1));
    units.push_back(std::make_unique<LoadUnitMock>(30, 0, unitsLoaded, 2));
    factory.create(std::move(units), skipToRow);
  };

  EXPECT_NO_THROW(getFactoryWithSkip(0));
  EXPECT_NO_THROW(getFactoryWithSkip(1));
  EXPECT_NO_THROW(getFactoryWithSkip(9));
  EXPECT_NO_THROW(getFactoryWithSkip(10));
  EXPECT_NO_THROW(getFactoryWithSkip(11));
  EXPECT_NO_THROW(getFactoryWithSkip(29));
  EXPECT_NO_THROW(getFactoryWithSkip(30));
  EXPECT_NO_THROW(getFactoryWithSkip(31));
  EXPECT_NO_THROW(getFactoryWithSkip(59));
  EXPECT_NO_THROW(getFactoryWithSkip(60));
  VELOX_ASSERT_THROW(
      getFactoryWithSkip(61),
      "Can only skip up to the past-the-end row of the file.");
  VELOX_ASSERT_THROW(
      getFactoryWithSkip(100),
      "Can only skip up to the past-the-end row of the file.");
}

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<
    UnitLoaderFactoryType>::testCanRequestUnitMultipleTimes() {
  auto factory = createFactory();
  std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

  auto unitLoader = factory.create(std::move(units), 0);
  unitLoader->getLoadedUnit(0);
  unitLoader->getLoadedUnit(0);
  unitLoader->getLoadedUnit(0);
}

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<UnitLoaderFactoryType>::testUnitOutOfRange() {
  auto factory = createFactory();
  std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

  auto unitLoader = factory.create(std::move(units), 0);
  unitLoader->getLoadedUnit(0);

  VELOX_ASSERT_THROW(unitLoader->getLoadedUnit(1), "Unit out of range");
}

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<UnitLoaderFactoryType>::testSeekOutOfRange() {
  auto factory = createFactory();
  std::vector<std::atomic_bool> unitsLoaded(getUnitsLoadedWithFalse(1));
  std::vector<std::unique_ptr<LoadUnit>> units;
  units.push_back(std::make_unique<LoadUnitMock>(10, 0, unitsLoaded, 0));

  auto unitLoader = factory.create(std::move(units), 0);

  unitLoader->onSeek(0, 10);

  VELOX_ASSERT_THROW(unitLoader->onSeek(0, 11), "Row out of range");
}

template <typename UnitLoaderFactoryType>
void UnitLoaderBaseTest<
    UnitLoaderFactoryType>::testSeekOutOfRangeReaderError() {
  auto factory = createFactory();
  ReaderMock readerMock{{10, 20, 30}, {0, 0, 0}, factory, 0};

  readerMock.seek(59);
  readerMock.seek(60);

  VELOX_ASSERT_THROW(
      readerMock.seek(61),
      "Can't seek to possition 61 in file. Must be up to 60.");
}
