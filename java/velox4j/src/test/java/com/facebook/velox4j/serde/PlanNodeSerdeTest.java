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

import org.junit.*;

import com.facebook.velox4j.Velox4j;
import com.facebook.velox4j.aggregate.Aggregate;
import com.facebook.velox4j.aggregate.AggregateStep;
import com.facebook.velox4j.connector.CommitStrategy;
import com.facebook.velox4j.data.BaseVectorTests;
import com.facebook.velox4j.expression.ConstantTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.join.JoinType;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.plan.AggregationNode;
import com.facebook.velox4j.plan.FilterNode;
import com.facebook.velox4j.plan.HashJoinNode;
import com.facebook.velox4j.plan.LimitNode;
import com.facebook.velox4j.plan.OrderByNode;
import com.facebook.velox4j.plan.PlanNode;
import com.facebook.velox4j.plan.ProjectNode;
import com.facebook.velox4j.plan.TableWriteNode;
import com.facebook.velox4j.plan.ValuesNode;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.sort.SortOrder;
import com.facebook.velox4j.test.Velox4jTests;
import com.facebook.velox4j.type.IntegerType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.variant.BooleanValue;

public class PlanNodeSerdeTest {
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
  public void testSortOrder() {
    final SortOrder order = new SortOrder(true, true);
    SerdeTests.testJavaBeanRoundTrip(order);
  }

  @Test
  public void testAggregateStep() {
    SerdeTests.testJavaBeanRoundTrip(AggregateStep.INTERMEDIATE);
  }

  @Test
  public void testAggregate() {
    final Aggregate aggregate = SerdeTests.newSampleAggregate();
    SerdeTests.testJavaBeanRoundTrip(aggregate);
  }

  @Test
  public void testJoinType() {
    SerdeTests.testJavaBeanRoundTrip(JoinType.LEFT_SEMI_FILTER);
  }

  // Ignored by https://github.com/velox4j/velox4j/issues/104.
  @Ignore
  public void testValuesNode() {
    // The case fails in debug build. Should investigate.
    final PlanNode values =
        ValuesNode.create("id-1", List.of(BaseVectorTests.newSampleRowVector(session)), true, 1);
    SerdeTests.testISerializableRoundTrip(values);
  }

  @Test
  public void testTableScanNode() {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode("id-1", SerdeTests.newSampleOutputType());
    SerdeTests.testISerializableRoundTrip(scan);
  }

  @Test
  public void testAggregationNode() {
    final AggregationNode aggregationNode = SerdeTests.newSampleAggregationNode("id-2", "id-1");
    SerdeTests.testISerializableRoundTrip(aggregationNode);
  }

  @Test
  public void testProjectNode() {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode("id-1", SerdeTests.newSampleOutputType());
    final ProjectNode projectNode =
        new ProjectNode(
            "id-2",
            List.of(scan),
            List.of("foo"),
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo")));
    SerdeTests.testISerializableRoundTrip(projectNode);
  }

  @Test
  public void testFilterNode() {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode("id-1", SerdeTests.newSampleOutputType());
    final FilterNode filterNode =
        new FilterNode("id-2", List.of(scan), ConstantTypedExpr.create(new BooleanValue(true)));
    SerdeTests.testISerializableRoundTrip(filterNode);
  }

  @Test
  public void testHashJoinNode() {
    final PlanNode scan1 =
        SerdeTests.newSampleTableScanNode(
            "id-1",
            new RowType(List.of("foo1", "bar1"), List.of(new IntegerType(), new IntegerType())));
    final PlanNode scan2 =
        SerdeTests.newSampleTableScanNode(
            "id-2",
            new RowType(List.of("foo2", "bar2"), List.of(new IntegerType(), new IntegerType())));
    final PlanNode joinNode =
        new HashJoinNode(
            "id-3",
            JoinType.INNER,
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo1")),
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo2")),
            ConstantTypedExpr.create(new BooleanValue(true)),
            scan1,
            scan2,
            new RowType(
                List.of("foo1", "bar1", "foo2", "bar2"),
                List.of(
                    new IntegerType(), new IntegerType(), new IntegerType(), new IntegerType())),
            false);
    SerdeTests.testISerializableRoundTrip(joinNode);
  }

  @Test
  public void testOrderByNode() {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode("id-1", SerdeTests.newSampleOutputType());
    final OrderByNode orderByNode =
        new OrderByNode(
            "id-2",
            List.of(scan),
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo1")),
            List.of(new SortOrder(true, false)),
            false);
    SerdeTests.testISerializableRoundTrip(orderByNode);
  }

  @Test
  public void testLimitNode() {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode("id-1", SerdeTests.newSampleOutputType());
    final LimitNode limitNode = new LimitNode("id-2", List.of(scan), 5, 3, false);
    SerdeTests.testISerializableRoundTrip(limitNode);
  }

  @Test
  public void testTableWriteNode() {
    final RowType rowType = SerdeTests.newSampleOutputType();
    final PlanNode scan = SerdeTests.newSampleTableScanNode("id-1", rowType);
    final TableWriteNode tableWriteNode =
        new TableWriteNode(
            "id-2",
            rowType,
            rowType.getNames(),
            SerdeTests.newSampleAggregationNode("id-4", "id-3"),
            "connector-1",
            SerdeTests.newSampleHiveInsertTableHandle(),
            true,
            rowType,
            CommitStrategy.TASK_COMMIT,
            List.of(scan));
    SerdeTests.testISerializableRoundTrip(tableWriteNode);
  }
}
