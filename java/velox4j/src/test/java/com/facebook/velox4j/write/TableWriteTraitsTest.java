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
package com.facebook.velox4j.write;

import org.junit.*;

import com.facebook.velox4j.Velox4j;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.serde.SerdeTests;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.test.ResourceTests;
import com.facebook.velox4j.test.Velox4jTests;
import com.facebook.velox4j.type.RowType;

public class TableWriteTraitsTest {
  private static MemoryManager memoryManager;
  private static Session session;

  @BeforeClass
  public static void beforeClass() throws Exception {
    Velox4jTests.ensureInitialized();
    memoryManager = MemoryManager.create(AllocationListener.NOOP);
  }

  @AfterClass
  public static void afterClass() throws Exception {
    memoryManager.close();
  }

  @Before
  public void setUp() throws Exception {
    session = Velox4j.newSession(memoryManager);
  }

  @After
  public void tearDown() throws Exception {
    session.close();
  }

  @Test
  public void testOutputType() {
    final RowType type = TableWriteTraits.outputType();
    Assert.assertEquals(
        ResourceTests.readResourceAsString("table-write-traits/output-type-1.json").stripTrailing(),
        Serde.toPrettyJson(type));
  }

  @Test
  public void testOutputTypeWithAggregationNode() {
    final RowType type =
        session
            .tableWriteTraitsOperations()
            .outputType(SerdeTests.newSampleAggregationNode("id-2", "id-1"));
    Assert.assertEquals(
        ResourceTests.readResourceAsString(
                "table-write-traits/output-type-with-aggregation-node-1.json")
            .stripTrailing(),
        Serde.toPrettyJson(type));
  }
}
