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
package com.facebook.velox4j.serde;

import java.util.List;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.test.Velox4jTests;
import com.facebook.velox4j.type.ArrayType;
import com.facebook.velox4j.type.BigIntType;
import com.facebook.velox4j.type.BooleanType;
import com.facebook.velox4j.type.DateType;
import com.facebook.velox4j.type.DecimalType;
import com.facebook.velox4j.type.DoubleType;
import com.facebook.velox4j.type.FunctionType;
import com.facebook.velox4j.type.HugeIntType;
import com.facebook.velox4j.type.IntegerType;
import com.facebook.velox4j.type.IntervalDayTimeType;
import com.facebook.velox4j.type.IntervalYearMonthType;
import com.facebook.velox4j.type.MapType;
import com.facebook.velox4j.type.OpaqueType;
import com.facebook.velox4j.type.RealType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.SmallIntType;
import com.facebook.velox4j.type.TimestampType;
import com.facebook.velox4j.type.TinyIntType;
import com.facebook.velox4j.type.UnknownType;
import com.facebook.velox4j.type.VarCharType;
import com.facebook.velox4j.type.VarbinaryType;

public class TypeSerdeTest {

  @BeforeClass
  public static void beforeClass() throws Exception {
    Velox4jTests.ensureInitialized();
  }

  @Test
  public void testBoolean() {
    SerdeTests.testISerializableRoundTrip(new BooleanType());
  }

  @Test
  public void testTinyInt() {
    SerdeTests.testISerializableRoundTrip(new TinyIntType());
  }

  @Test
  public void testSmallInt() {
    SerdeTests.testISerializableRoundTrip(new SmallIntType());
  }

  @Test
  public void testInteger() {
    SerdeTests.testISerializableRoundTrip(new IntegerType());
  }

  @Test
  public void testBigInt() {
    SerdeTests.testISerializableRoundTrip(new BigIntType());
  }

  @Test
  public void testHugeInt() {
    SerdeTests.testISerializableRoundTrip(new HugeIntType());
  }

  @Test
  public void testRealType() {
    SerdeTests.testISerializableRoundTrip(new RealType());
  }

  @Test
  public void testDoubleType() {
    SerdeTests.testISerializableRoundTrip(new DoubleType());
  }

  @Test
  public void testVarcharType() {
    SerdeTests.testISerializableRoundTrip(new VarCharType());
  }

  @Test
  public void testVarbinaryType() {
    SerdeTests.testISerializableRoundTrip(new VarbinaryType());
  }

  @Test
  public void testTimestampType() {
    SerdeTests.testISerializableRoundTrip(new TimestampType());
  }

  @Test
  public void testArrayType() {
    SerdeTests.testISerializableRoundTrip(ArrayType.create(new IntegerType()));
  }

  @Test
  public void testMapType() {
    SerdeTests.testISerializableRoundTrip(MapType.create(new IntegerType(), new VarCharType()));
  }

  @Test
  public void testRowType() {
    SerdeTests.testISerializableRoundTrip(
        new RowType(List.of("foo", "bar"), List.of(new IntegerType(), new VarCharType())));
  }

  @Test
  public void testFunctionType() {
    SerdeTests.testISerializableRoundTrip(
        FunctionType.create(List.of(new IntegerType(), new VarCharType()), new VarbinaryType()));
  }

  @Test
  public void testUnknownType() {
    SerdeTests.testISerializableRoundTrip(new UnknownType());
  }

  @Test
  public void testOpaqueType() {
    Assert.assertThrows(
        VeloxException.class, () -> SerdeTests.testISerializableRoundTrip(new OpaqueType("foo")));
  }

  @Test
  public void testDecimalType() {
    SerdeTests.testISerializableRoundTrip(new DecimalType(10, 5));
  }

  @Test
  public void testIntervalDayTimeType() {
    SerdeTests.testISerializableRoundTrip(new IntervalDayTimeType());
  }

  @Test
  public void testIntervalYearMonthType() {
    SerdeTests.testISerializableRoundTrip(new IntervalYearMonthType());
  }

  @Test
  public void testDateType() {
    SerdeTests.testISerializableRoundTrip(new DateType());
  }
}
