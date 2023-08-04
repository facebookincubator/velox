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

#include "velox/dwio/dwrf/test/TestColumnStatisticsBase.h"

using namespace facebook::velox::dwrf;

TEST(StatisticsBuilder, size) {
  testSize();
}

TEST(StatisticsBuilder, integer) {
  testInteger();
}

TEST(StatisticsBuilder, integerMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  auto intProto = proto.mutable_intstatistics();
  testIntegerMissingStats(columnStatisticsWrapper, intProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, integerEmptyStats) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(0);
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testIntegerEmptyStats(
      columnStatisticsWrapper, (void*)&proto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, integerOverflow) {
  testIntegerOverflow();
}

TEST(StatisticsBuilder, doubles) {
  testDoubles();
}

TEST(StatisticsBuilder, doubleMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  auto doubleProto = proto.mutable_doublestatistics();
  testDoubleMissingStats(
      columnStatisticsWrapper, doubleProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, doubleEmptyStats) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(0);
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testDoubleEmptyStats(
      columnStatisticsWrapper, (void*)&proto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, doubleNaN) {
  testDoubleNaN();
}

TEST(StatisticsBuilder, string) {
  testString();
}

TEST(StatisticsBuilder, stringMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  auto strProto = proto.mutable_stringstatistics();
  testStringMissingStats(columnStatisticsWrapper, strProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, stringEmptyStats) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(0);
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testStringEmptyStats(
      columnStatisticsWrapper, (void*)&proto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, stringLengthThreshold) {
  testStringLengthThreshold();
}

TEST(StatisticsBuilder, stringLengthOverflow) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(1);
  auto strProto = proto.mutable_stringstatistics();
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testStringLengthOverflow(columnStatisticsWrapper, strProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, boolean) {
  testBoolean();
}

TEST(StatisticsBuilder, booleanMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto boolProto = proto.mutable_bucketstatistics();
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBooleanMissingStats(columnStatisticsWrapper, boolProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, booleanEmptyStats) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(0);
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBooleanEmptyStats(
      columnStatisticsWrapper, (void*)&proto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, basic) {
  testBasic();
}

TEST(StatisticsBuilder, basicMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBasicMissingStats(columnStatisticsWrapper);
}

TEST(StatisticsBuilder, basicHasNull) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBasicHasNull(columnStatisticsWrapper, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, binary) {
  testBinary();
}

TEST(StatisticsBuilder, binaryMissingStats) {
  proto::orc::ColumnStatistics proto;
  auto binProto = proto.mutable_binarystatistics();
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBinaryMissingStats(columnStatisticsWrapper, binProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, binaryEmptyStats) {
  proto::orc::ColumnStatistics proto;
  proto.set_numberofvalues(0);
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBinaryEmptyStats(
      columnStatisticsWrapper, (void*)&proto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, binaryLengthOverflow) {
  proto::orc::ColumnStatistics proto;
  auto binProto = proto.mutable_binarystatistics();
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  testBinaryLengthOverflow(columnStatisticsWrapper, binProto, DwrfFormat::kOrc);
}

TEST(StatisticsBuilder, initialSize) {
  testInitialSize();
}

TEST(MapStatistics, orcUnsupportedMapStatistics) {
  proto::orc::ColumnStatistics proto;
  auto columnStatisticsWrapper = ColumnStatisticsWrapper(&proto);
  ASSERT_FALSE(columnStatisticsWrapper.hasMapStatistics());
  ASSERT_THROW(
      columnStatisticsWrapper.mapStatistics(),
      ::facebook::velox::VeloxRuntimeError);
}
