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

import java.util.OptionalLong;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import com.facebook.velox4j.connector.*;
import com.facebook.velox4j.filter.AlwaysTrue;
import com.facebook.velox4j.test.Velox4jTests;

public class ConnectorSerdeTest {
  @BeforeClass
  public static void beforeClass() {
    Velox4jTests.ensureInitialized();
  }

  @Test
  public void testFileFormat() {
    final FileFormat in = FileFormat.DWRF;
    final String json = SerdeTests.testJavaBeanRoundTrip(in).getJson();
    Assert.assertEquals("\"dwrf\"", json);
  }

  @Test
  public void testProperties() {
    final FileProperties in = new FileProperties(OptionalLong.of(100), OptionalLong.of(50));
    SerdeTests.testJavaBeanRoundTrip(in);
  }

  @Test
  public void testPropertiesWithMissingFields() {
    final FileProperties in = new FileProperties(OptionalLong.of(100), OptionalLong.empty());
    SerdeTests.testJavaBeanRoundTrip(in);
  }

  @Test
  public void testSubfieldFilter() {
    final SubfieldFilter in = new SubfieldFilter("complex_type[1][\"foo\"].id", new AlwaysTrue());
    SerdeTests.testJavaBeanRoundTrip(in);
  }

  @Test
  public void testAssignment() {
    final Assignment assignment = new Assignment("foo", SerdeTests.newSampleHiveColumnHandle());
    SerdeTests.testJavaBeanRoundTrip(assignment);
  }

  @Test
  public void testRowIdProperties() {
    final RowIdProperties in = new RowIdProperties(5, 10, "UUID-100");
    SerdeTests.testJavaBeanRoundTrip(in);
  }

  @Test
  public void testHiveColumnHandle() {
    final ColumnHandle handle = SerdeTests.newSampleHiveColumnHandle();
    SerdeTests.testISerializableRoundTrip(handle);
  }

  @Test
  public void testHiveConnectorSplit() {
    final ConnectorSplit split = SerdeTests.newSampleHiveSplit();
    SerdeTests.testISerializableRoundTrip(split);
  }

  @Test
  public void testHiveConnectorSplitWithMissingFields() {
    final ConnectorSplit split = SerdeTests.newSampleHiveSplitWithMissingFields();
    SerdeTests.testISerializableRoundTrip(split);
  }

  @Test
  public void testHiveTableHandle() {
    final ConnectorTableHandle handle =
        SerdeTests.newSampleHiveTableHandle(SerdeTests.newSampleOutputType());
    SerdeTests.testISerializableRoundTrip(handle);
  }

  @Test
  public void testExternalStreamConnectorSplit() {
    final ConnectorSplit split = new ExternalStreamConnectorSplit("id-1", 100);
    SerdeTests.testISerializableRoundTrip(split);
  }

  @Test
  public void testExternalStreamTableHandle() {
    final ExternalStreamTableHandle handle = new ExternalStreamTableHandle("id-1");
    SerdeTests.testISerializableRoundTrip(handle);
  }

  @Test
  public void testLocationHandle() {
    final LocationHandle handle = SerdeTests.newSampleLocationHandle();
    SerdeTests.testISerializableRoundTrip(handle);
  }

  @Test
  public void testHiveBucketProperty() {
    final HiveBucketProperty hiveBucketProperty = SerdeTests.newSampleHiveBucketProperty();
    SerdeTests.testISerializableRoundTrip(hiveBucketProperty);
  }

  @Test
  public void testHiveInsertTableHandle() {
    final HiveInsertTableHandle hiveInsertTableHandle = SerdeTests.newSampleHiveInsertTableHandle();
    SerdeTests.testISerializableRoundTrip(hiveInsertTableHandle);
  }

  @Test
  public void testHiveInsertFileNameGenerator() {
    final HiveInsertFileNameGenerator fileNameGenerator = new HiveInsertFileNameGenerator();
    SerdeTests.testISerializableRoundTrip(fileNameGenerator);
  }
}
