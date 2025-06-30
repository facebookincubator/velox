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
package com.facebook.velox4j.query;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.JsonNode;
import org.junit.*;

import com.facebook.velox4j.Velox4j;
import com.facebook.velox4j.aggregate.Aggregate;
import com.facebook.velox4j.aggregate.AggregateStep;
import com.facebook.velox4j.collection.Streams;
import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.conf.ConnectorConfig;
import com.facebook.velox4j.connector.*;
import com.facebook.velox4j.data.BaseVectorTests;
import com.facebook.velox4j.data.RowVector;
import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.expression.ConstantTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.iterator.*;
import com.facebook.velox4j.jni.JniWorkspace;
import com.facebook.velox4j.join.JoinType;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.plan.*;
import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.sort.SortOrder;
import com.facebook.velox4j.test.*;
import com.facebook.velox4j.test.dataset.TestDataFile;
import com.facebook.velox4j.test.dataset.tpch.TpchDatasets;
import com.facebook.velox4j.test.dataset.tpch.TpchTableName;
import com.facebook.velox4j.type.BigIntType;
import com.facebook.velox4j.type.BooleanType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;
import com.facebook.velox4j.type.VarCharType;
import com.facebook.velox4j.variant.BigIntValue;
import com.facebook.velox4j.variant.BooleanValue;
import com.facebook.velox4j.write.TableWriteTraits;

public class QueryTest {
  private static final String HIVE_CONNECTOR_ID = "connector-hive";
  private static TestDataFile NATION_FILE;
  private static TestDataFile REGION_FILE;
  private static MemoryManager memoryManager;
  private static Session session;

  @BeforeClass
  public static void beforeClass() throws Exception {
    Velox4jTests.ensureInitialized();
    memoryManager = MemoryManager.create(AllocationListener.NOOP);
    NATION_FILE = TpchDatasets.get().get(TpchTableName.NATION);
    REGION_FILE = TpchDatasets.get().get(TpchTableName.REGION);
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
  public void testValuesWith2Steps() {
    final PlanNode values =
        ValuesNode.create("id-1", List.of(BaseVectorTests.newSampleRowVector(session)), true, 5);
    final Query query = new Query(values, Config.empty(), ConnectorConfig.empty());
    final QueryExecutor exec = session.queryOperations().createQueryExecutor(query);
    final SerialTask task = exec.execute();
    SampleQueryTests.assertIterator(task, 5);
  }

  @Test
  public void testValues() {
    final PlanNode values =
        ValuesNode.create("id-1", List.of(BaseVectorTests.newSampleRowVector(session)), true, 5);
    final Query query = new Query(values, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    SampleQueryTests.assertIterator(task, 5);
  }

  @Test
  public void testTableScan1() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-table-scan-nation.tsv"))
        .run();
  }

  @Test
  public void testTableScan2() {
    final File file = REGION_FILE.file();
    final RowType outputType = REGION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-table-scan-region.tsv"))
        .run();
  }

  @Test
  public void testTableScanCollectMultipleRowVectorsLoadInline() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final int maxOutputBatchRows = 7;
    final Query query =
        new Query(
            scanNode,
            Config.create(Map.of("max_output_batch_rows", String.format("%d", maxOutputBatchRows))),
            ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final List<RowVector> allRvs =
        Streams.fromIterator(UpIterators.asJavaIterator(task))
            .map(v -> v.loadedVector().asRowVector())
            .collect(Collectors.toList());
    Assert.assertTrue(allRvs.size() > 1);
    for (RowVector rv : allRvs) {
      Assert.assertTrue(rv.getSize() <= maxOutputBatchRows);
    }
    final RowVector appended =
        session.baseVectorOperations().createEmpty(allRvs.get(0).getType()).asRowVector();
    for (RowVector rv : allRvs) {
      appended.append(rv);
    }
    Assert.assertEquals(
        ResourceTests.readResourceAsString("query-output/tpch-table-scan-nation.tsv"),
        appended.toString());
  }

  @Test
  public void testTableScanCollectMultipleRowVectorsLoadLast() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final int maxOutputBatchRows = 7;
    final Query query =
        new Query(
            scanNode,
            Config.create(Map.of("max_output_batch_rows", String.format("%d", maxOutputBatchRows))),
            ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final List<RowVector> allRvs =
        Streams.fromIterator(UpIterators.asJavaIterator(task)).collect(Collectors.toList());
    Assert.assertTrue(allRvs.size() > 1);
    for (int i = 0; i < allRvs.size(); i++) {
      final RowVector rv = allRvs.get(i);
      Assert.assertTrue(rv.getSize() <= maxOutputBatchRows);
      if (i != allRvs.size() - 1) {
        // Vectors except the last one should throw when loading.
        Assert.assertThrows(VeloxException.class, rv::loadedVector);
      } else {
        // The last vector can be loaded without errors.
        rv.loadedVector();
      }
    }
  }

  @Test
  public void testAggregate() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    ;
    final ConnectorSplit split = newSampleSplit(file);
    final AggregationNode aggregationNode =
        newSampleAggregationNodeSumNationKeyByRegionKey("id-2", scanNode);
    final Query query = new Query(aggregationNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-aggregate-1.tsv"))
        .run();
  }

  @Test
  public void testAggregateStats() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    ;
    final ConnectorSplit split = newSampleSplit(file);
    final AggregationNode aggregationNode =
        newSampleAggregationNodeSumNationKeyByRegionKey("id-2", scanNode);
    final Query query = new Query(aggregationNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask serialTask = session.queryOperations().execute(query);
    serialTask.addSplit(scanNode.getId(), split);
    serialTask.noMoreSplits(scanNode.getId());
    UpIteratorTests.collect(serialTask);
    final SerialTaskStats serialTaskStats = serialTask.collectStats();

    final JsonNode scanStats = serialTaskStats.planStats("id-1");
    Assert.assertEquals("TableScan", scanStats.get("operatorType").asText());
    Assert.assertEquals(1, scanStats.get("numDrivers").asInt());
    Assert.assertEquals(1, scanStats.get("numSplits").asInt());
    Assert.assertEquals(25, scanStats.get("inputRows").asInt());

    final JsonNode aggStats = serialTaskStats.planStats("id-2");
    Assert.assertEquals("Aggregation", aggStats.get("operatorType").asText());
    Assert.assertEquals(25, aggStats.get("inputRows").asInt());
    Assert.assertEquals(5, aggStats.get("outputRows").asInt());
  }

  @Test
  public void testExternalStreamFromJavaIterator() {
    final String json = SampleQueryTests.readQueryJson();
    final UpIterator sampleIn =
        session.queryOperations().execute(Serde.fromJson(json, Query.class));
    final DownIterator down = DownIterators.fromJavaIterator(UpIterators.asJavaIterator(sampleIn));
    final ExternalStream es = session.externalStreamOperations().bind(down);
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    ;
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", es.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    SampleQueryTests.assertIterator(task);
  }

  @Test
  public void testBlockingQueue() throws InterruptedException {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final RowVector rv = BaseVectorTests.newSampleRowVector(session);

    // No input added, the up-iterator is considered blocked.
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add one input.
    queue.put(rv);
    task.waitFor();
    Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
    Assert.assertThrows(VeloxException.class, task::advance);
    BaseVectorTests.assertEquals(rv, task.get());
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add multiple inputs at a time.
    queue.put(rv);
    queue.put(rv);
    task.waitFor();
    Assert.assertThrows(VeloxException.class, task::waitFor);
    Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
    Assert.assertThrows(VeloxException.class, task::advance);
    BaseVectorTests.assertEquals(rv, task.get());
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
    Assert.assertThrows(VeloxException.class, task::advance);
    BaseVectorTests.assertEquals(rv, task.get());
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
  }

  @Test
  public void testBlockingQueueNoMoreInput() throws InterruptedException {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final RowVector rv = BaseVectorTests.newSampleRowVector(session);

    // No input added, the up-iterator is considered blocked.
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add one input.
    queue.put(rv);
    task.waitFor();
    Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
    Assert.assertThrows(VeloxException.class, task::advance);
    BaseVectorTests.assertEquals(rv, task.get());
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add another input, then signal no-more-input.
    queue.put(rv);
    queue.noMoreInput();
    task.waitFor();
    Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
    Assert.assertThrows(VeloxException.class, task::advance);
    BaseVectorTests.assertEquals(rv, task.get());
    Assert.assertEquals(UpIterator.State.FINISHED, task.advance());
    Assert.assertThrows(VeloxException.class, task::get);
  }

  @Test
  public void testBlockingQueueNoMoreInputTwoThreads() throws InterruptedException {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());

    final Thread consumer =
        TestThreads.newTestThread(
            () -> {
              while (true) {
                final UpIterator.State state = task.advance();
                if (state == UpIterator.State.BLOCKED) {
                  task.waitFor();
                  continue;
                }
                Assert.assertEquals(UpIterator.State.FINISHED, state);
                Assert.assertThrows(VeloxException.class, task::get);
                break;
              }
            });
    consumer.start();

    Thread.sleep(500L);
    queue.noMoreInput();
    consumer.join();
  }

  @Test
  public void testBlockingQueueWithInfiniteIteratorOut() throws Exception {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());

    final InfiniteIterator<RowVector> out;
    final RowVector rv = BaseVectorTests.newSampleRowVector(session);

    {
      final SerialTask task = session.queryOperations().execute(query);
      task.addSplit(scanNode.getId(), split);
      task.noMoreSplits(scanNode.getId());
      out = UpIterators.asInfiniteIterator(task);
    }

    // No input added, the iterator is not available.
    Assert.assertThrows(VeloxException.class, out::get);
    Assert.assertFalse(out.available());
    Assert.assertFalse(out.available());

    // Add one input.
    queue.put(rv);
    Assert.assertThrows(VeloxException.class, out::get);
    out.waitFor();
    out.waitFor();
    Assert.assertTrue(out.available());
    BaseVectorTests.assertEquals(rv, out.get());
    Assert.assertThrows(VeloxException.class, out::get);
    Assert.assertFalse(out.available());
    Assert.assertFalse(out.available());

    // Add multiple inputs at a time.
    queue.put(rv);
    queue.put(rv);
    Assert.assertThrows(VeloxException.class, out::get);
    out.waitFor();
    Assert.assertTrue(out.available());
    Assert.assertTrue(out.available());
    out.waitFor();
    BaseVectorTests.assertEquals(rv, out.get());
    Assert.assertThrows(VeloxException.class, out::get);
    out.waitFor();
    Assert.assertTrue(out.available());
    Assert.assertTrue(out.available());
    out.waitFor();
    BaseVectorTests.assertEquals(rv, out.get());
    Assert.assertThrows(VeloxException.class, out::get);
    Assert.assertFalse(out.available());
    Assert.assertFalse(out.available());

    // Async wait.
    final Thread asyncWaiter =
        TestThreads.newTestThread(
            new Runnable() {
              @Override
              public void run() {
                out.waitFor();
                Assert.assertTrue(out.available());
                BaseVectorTests.assertEquals(rv, out.get());
                Assert.assertFalse(out.available());
              }
            });
    asyncWaiter.start();
    Thread.sleep(500L);
    queue.put(rv);
    asyncWaiter.join();
  }

  @Test
  public void testBlockingQueueTwoThreads() throws InterruptedException {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(scanNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final RowVector rv = BaseVectorTests.newSampleRowVector(session);

    final Object control = new Object();

    // No input added, the up-iterator is considered blocked.
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    final Thread testThread =
        TestThreads.newTestThread(
            () -> {
              try {
                synchronized (control) {
                  // Signals the main thread to add one input after 500ms.
                  control.notifyAll();
                  control.wait();

                  // The wait calls should not throw.
                  task.waitFor();
                  Assert.assertThrows(VeloxException.class, task::waitFor);
                  Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
                  Assert.assertThrows(VeloxException.class, task::advance);
                  BaseVectorTests.assertEquals(rv, task.get());
                  Assert.assertThrows(VeloxException.class, task::get);
                  Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
                  Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

                  // Signals the main thread to add two inputs after 500ms.
                  control.notifyAll();
                  control.wait();

                  // The wait calls should not throw.
                  task.waitFor();
                  Assert.assertThrows(VeloxException.class, task::waitFor);
                  Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
                  Assert.assertThrows(VeloxException.class, task::advance);
                  BaseVectorTests.assertEquals(rv, task.get());
                  Assert.assertThrows(VeloxException.class, task::get);
                  Assert.assertEquals(UpIterator.State.AVAILABLE, task.advance());
                  Assert.assertThrows(VeloxException.class, task::advance);
                  BaseVectorTests.assertEquals(rv, task.get());
                  Assert.assertThrows(VeloxException.class, task::get);
                  Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
                  Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

                  // Signals the main thread that the test has passed.
                  control.notifyAll();
                }
              } catch (InterruptedException e) {
                throw new RuntimeException(e);
              }
            });

    synchronized (control) {
      // This makes sure test thread starts processing after control.wait().
      testThread.start();
      control.wait();

      // Add one input after 1s.
      TestThreads.newTestThread(
              () -> {
                try {
                  Thread.sleep(500L);
                  queue.put(rv);
                } catch (InterruptedException e) {
                  throw new RuntimeException(e);
                }
              })
          .start();

      // Signals the test thread to start checking the output.
      control.notifyAll();
      control.wait();

      // Add two inputs at a time after 1s.
      TestThreads.newTestThread(
              () -> {
                try {
                  Thread.sleep(500L);
                  queue.put(rv);
                  queue.put(rv);
                } catch (InterruptedException e) {
                  throw new RuntimeException(e);
                }
              })
          .start();

      // Signals the test thread to start checking the output.
      control.notifyAll();
      control.wait();
    }

    testThread.join();
  }

  @Test
  public void testBlockingQueueWithInputFiltered() throws InterruptedException {
    final ExternalStreams.BlockingQueue queue =
        session.externalStreamOperations().newBlockingQueue();
    final TableScanNode scanNode =
        new TableScanNode(
            "id-1",
            SampleQueryTests.getSchema(),
            new ExternalStreamTableHandle("connector-external-stream"),
            List.of());
    final FilterNode filterNode =
        new FilterNode(
            "id-2", List.of(scanNode), ConstantTypedExpr.create(new BooleanValue(false)));
    final ConnectorSplit split =
        new ExternalStreamConnectorSplit("connector-external-stream", queue.id());
    final Query query = new Query(filterNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    final RowVector rv = BaseVectorTests.newSampleRowVector(session);

    // No input added, the up-iterator is considered blocked.
    Assert.assertThrows(VeloxException.class, task::get);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add one input.
    queue.put(rv);
    Thread.sleep(500L);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());

    // Add multiple inputs at a time.
    queue.put(rv);
    queue.put(rv);
    Thread.sleep(500L);
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
    Assert.assertEquals(UpIterator.State.BLOCKED, task.advance());
  }

  @Test
  public void testProject() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final ProjectNode projectNode =
        new ProjectNode(
            "id-2",
            List.of(scanNode),
            List.of("n_nationkey", "n_comment"),
            List.of(
                FieldAccessTypedExpr.create(new BigIntType(), "n_nationkey"),
                FieldAccessTypedExpr.create(new VarCharType(), "n_comment")));
    final Query query = new Query(projectNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-project-1.tsv"))
        .run();
  }

  @Test
  public void testFilter() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final FilterNode filterNode =
        new FilterNode(
            "id-2",
            List.of(scanNode),
            new CallTypedExpr(
                new BooleanType(),
                List.of(
                    FieldAccessTypedExpr.create(new BigIntType(), "n_regionkey"),
                    ConstantTypedExpr.create(new BigIntValue(3L))),
                "greaterthanorequal"));
    final Query query = new Query(filterNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-filter-1.tsv"))
        .run();
  }

  @Test
  public void testHashJoin() {
    final File nationFile = NATION_FILE.file();
    final RowType nationOutputType = NATION_FILE.schema();
    final File regionFile = REGION_FILE.file();
    final RowType regionOutputType = REGION_FILE.schema();
    final TableScanNode nationScanNode = newSampleTableScanNode("id-1", nationOutputType);
    final TableScanNode regionScanNode = newSampleTableScanNode("id-2", regionOutputType);
    final ConnectorSplit nationSplit = newSampleSplit(nationFile);
    final ConnectorSplit regionSplit = newSampleSplit(regionFile);
    final HashJoinNode hashJoinNode =
        new HashJoinNode(
            "id-3",
            JoinType.LEFT,
            List.of(FieldAccessTypedExpr.create(new BigIntType(), "n_regionkey")),
            List.of(FieldAccessTypedExpr.create(new BigIntType(), "r_regionkey")),
            null,
            nationScanNode,
            regionScanNode,
            new RowType(
                List.of("n_nationkey", "n_name", "r_regionkey", "r_name"),
                List.of(new BigIntType(), new VarCharType(), new BigIntType(), new VarCharType())),
            false);
    final Query query = new Query(hashJoinNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(nationScanNode.getId(), nationSplit);
    task.addSplit(regionScanNode.getId(), regionSplit);
    task.noMoreSplits(nationScanNode.getId());
    task.noMoreSplits(regionScanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-join-1.tsv"))
        .run();
  }

  @Test
  public void testOrderBy() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final OrderByNode orderByNode =
        new OrderByNode(
            "id-2",
            List.of(scanNode),
            List.of(
                FieldAccessTypedExpr.create(new BigIntType(), "n_regionkey"),
                FieldAccessTypedExpr.create(new BigIntType(), "n_nationkey")),
            List.of(new SortOrder(true, false), new SortOrder(false, false)),
            false);
    final Query query = new Query(orderByNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-orderby-1.tsv"))
        .run();
  }

  @Test
  public void testLimit() {
    final File file = NATION_FILE.file();
    final RowType outputType = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", outputType);
    final ConnectorSplit split = newSampleSplit(file);
    final LimitNode limitNode = new LimitNode("id-2", List.of(scanNode), 5, 3, false);
    final Query query = new Query(limitNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-limit-1.tsv"))
        .run();
  }

  @Test
  public void testTableWrite() throws IOException {
    final File folder = JniWorkspace.getDefault().getSubDir("test");
    final String fileName = String.format("test-write-%s.tmp", UUID.randomUUID());
    final File file = NATION_FILE.file();
    final RowType schema = NATION_FILE.schema();
    final TableScanNode scanNode = newSampleTableScanNode("id-1", schema);
    final ConnectorSplit split = newSampleSplit(file);
    final TableWriteNode tableWriteNode =
        newSampleTableWriteNode("id-2", schema, folder, fileName, scanNode);
    final Query query = new Query(tableWriteNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task = session.queryOperations().execute(query);
    task.addSplit(scanNode.getId(), split);
    task.noMoreSplits(scanNode.getId());
    UpIteratorTests.assertIterator(task)
        .assertNumRowVectors(1)
        .assertRowVectorTypeJson(
            0,
            ResourceTests.readResourceAsString("query-output-type/tpch-table-write-1.json")
                .stripTrailing())
        .run();
  }

  @Test
  public void testTableWriteRoundTrip() throws IOException {
    final File folder = JniWorkspace.getDefault().getSubDir("test");
    final String fileName = String.format("test-write-%s.tmp", UUID.randomUUID());
    final RowType schema = NATION_FILE.schema();

    // Read the sample nation file.
    final File file = NATION_FILE.file();
    final TableScanNode scanNode1 = newSampleTableScanNode("id-1", schema);
    final ConnectorSplit split1 = newSampleSplit(file);
    final TableWriteNode tableWriteNode =
        newSampleTableWriteNode("id-2", schema, folder, fileName, scanNode1);
    final Query query1 = new Query(tableWriteNode, Config.empty(), ConnectorConfig.empty());
    final SerialTask task1 = session.queryOperations().execute(query1);
    task1.addSplit(scanNode1.getId(), split1);
    task1.noMoreSplits(scanNode1.getId());
    UpIteratorTests.assertIterator(task1)
        .assertNumRowVectors(1)
        .assertRowVectorTypeJson(
            0,
            ResourceTests.readResourceAsString("query-output-type/tpch-table-write-1.json")
                .stripTrailing())
        .run();

    // Read the file we just wrote.
    final File writtenFile = folder.toPath().resolve(fileName).toFile();
    ;
    final TableScanNode scanNode2 = newSampleTableScanNode("id-1", schema);
    final ConnectorSplit splits2 = newSampleSplit(writtenFile);
    final Query query2 = new Query(scanNode2, Config.empty(), ConnectorConfig.empty());
    final SerialTask task2 = session.queryOperations().execute(query2);
    task2.addSplit(scanNode2.getId(), splits2);
    task2.noMoreSplits(scanNode2.getId());
    UpIteratorTests.assertIterator(task2)
        .assertNumRowVectors(1)
        .assertRowVectorToString(
            0, ResourceTests.readResourceAsString("query-output/tpch-table-scan-nation.tsv"))
        .run();
  }

  private static TableWriteNode newSampleTableWriteNode(
      String id, RowType schema, File folder, String fileName, TableScanNode scanNode) {
    final ConnectorInsertTableHandle handle =
        new HiveInsertTableHandle(
            toColumnHandles(schema),
            new LocationHandle(
                folder.getAbsolutePath(),
                folder.getAbsolutePath(),
                LocationHandle.TableType.NEW,
                fileName),
            FileFormat.PARQUET,
            null,
            CompressionKind.GZIP,
            Map.of(),
            true,
            new HiveInsertFileNameGenerator());
    final RowType outputType = TableWriteTraits.outputType();
    final TableWriteNode tableWriteNode =
        new TableWriteNode(
            id,
            schema,
            schema.getNames(),
            null,
            HIVE_CONNECTOR_ID,
            handle,
            false,
            outputType,
            CommitStrategy.NO_COMMIT,
            List.of(scanNode));
    return tableWriteNode;
  }

  private static List<Assignment> toAssignments(RowType rowType) {
    final List<Assignment> list = new ArrayList<>();
    for (int i = 0; i < rowType.size(); i++) {
      final String name = rowType.getNames().get(i);
      final Type type = rowType.getChildren().get(i);
      list.add(
          new Assignment(
              name, new HiveColumnHandle(name, ColumnType.REGULAR, type, type, List.of())));
    }
    return list;
  }

  private static List<HiveColumnHandle> toColumnHandles(RowType rowType) {
    final List<HiveColumnHandle> list = new ArrayList<>();
    for (int i = 0; i < rowType.size(); i++) {
      final String name = rowType.getNames().get(i);
      final Type type = rowType.getChildren().get(i);
      list.add(new HiveColumnHandle(name, ColumnType.REGULAR, type, type, List.of()));
    }
    return list;
  }

  private static ConnectorSplit newSampleSplit(File file) {
    return new HiveConnectorSplit(
        "connector-hive",
        0,
        false,
        file.getAbsolutePath(),
        FileFormat.PARQUET,
        0,
        file.length(),
        Map.of(),
        OptionalInt.empty(),
        Optional.empty(),
        Map.of(),
        Optional.empty(),
        Map.of(),
        Map.of(),
        Optional.empty(),
        Optional.empty());
  }

  private static TableScanNode newSampleTableScanNode(String planNodeId, RowType outputType) {
    final TableScanNode scanNode =
        new TableScanNode(
            planNodeId,
            outputType,
            new HiveTableHandle(
                "connector-hive", "tab-1", false, List.of(), null, outputType, Map.of()),
            toAssignments(outputType));
    return scanNode;
  }

  private static AggregationNode newSampleAggregationNodeSumNationKeyByRegionKey(
      String planNodeId, PlanNode source) {
    final AggregationNode aggregationNode =
        new AggregationNode(
            planNodeId,
            AggregateStep.SINGLE,
            List.of(FieldAccessTypedExpr.create(new BigIntType(), "n_regionkey")),
            List.of(),
            List.of("cnt"),
            List.of(
                new Aggregate(
                    new CallTypedExpr(
                        new BigIntType(),
                        List.of(FieldAccessTypedExpr.create(new BigIntType(), "n_nationkey")),
                        "sum"),
                    List.of(new BigIntType()),
                    null,
                    List.of(),
                    List.of(),
                    false)),
            false,
            List.of(source),
            null,
            List.of());
    return aggregationNode;
  }
}
