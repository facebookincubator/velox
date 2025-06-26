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

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.ComparisonFailure;

import com.facebook.velox4j.aggregate.Aggregate;
import com.facebook.velox4j.aggregate.AggregateStep;
import com.facebook.velox4j.connector.*;
import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.filter.AlwaysTrue;
import com.facebook.velox4j.jni.JniApiTests;
import com.facebook.velox4j.jni.LocalSession;
import com.facebook.velox4j.memory.AllocationListener;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.plan.AggregationNode;
import com.facebook.velox4j.plan.PlanNode;
import com.facebook.velox4j.plan.TableScanNode;
import com.facebook.velox4j.serializable.ISerializable;
import com.facebook.velox4j.serializable.ISerializableCpp;
import com.facebook.velox4j.sort.SortOrder;
import com.facebook.velox4j.type.ArrayType;
import com.facebook.velox4j.type.BigIntType;
import com.facebook.velox4j.type.BooleanType;
import com.facebook.velox4j.type.IntegerType;
import com.facebook.velox4j.type.MapType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;
import com.facebook.velox4j.type.VarCharType;
import com.facebook.velox4j.variant.Variant;
import com.facebook.velox4j.variant.VariantCpp;

public final class SerdeTests {
  private static void assertJsonEquals(String expected, String actual) {
    final JsonNode expectedTree = Serde.parseTree(expected);
    final JsonNode actualTree = Serde.parseTree(actual);
    if (!actualTree.equals(expectedTree)) {
      throw new ComparisonFailure("", expected, actual);
    }
  }

  public static <T extends ISerializable> ObjectAndJson<ISerializable> testISerializableRoundTrip(
      T inObj) {
    try (final MemoryManager memoryManager = MemoryManager.create(AllocationListener.NOOP);
        final LocalSession session = JniApiTests.createLocalSession(memoryManager)) {
      final String inJson = Serde.toPrettyJson(inObj);

      {
        final ISerializable javaOutObj = Serde.fromJson(inJson, ISerializable.class);
        final String javaOutJson = Serde.toPrettyJson(javaOutObj);
        assertJsonEquals(inJson, javaOutJson);
      }

      try (final ISerializableCpp inObjCo = session.iSerializableOperations().asCpp(inObj)) {
        final ISerializable cppOutObj = inObjCo.asJava();
        final String cppOutJson = Serde.toPrettyJson(cppOutObj);
        assertJsonEquals(inJson, cppOutJson);
        return new ObjectAndJson<>(cppOutObj, cppOutJson);
      }
    }
  }

  public static <T extends ISerializable> ObjectAndJson<ISerializable> testISerializableRoundTrip(
      String inJson, Class<? extends T> valueType) {
    final T inObj = Serde.fromJson(inJson, valueType);
    return SerdeTests.testISerializableRoundTrip(inObj);
  }

  public static <T extends Variant> ObjectAndJson<Variant> testVariantRoundTrip(T inObj) {
    try (final MemoryManager memoryManager = MemoryManager.create(AllocationListener.NOOP);
        final LocalSession session = JniApiTests.createLocalSession(memoryManager)) {
      final String inJson = Serde.toPrettyJson(inObj);

      {
        final Variant javaOutObj = Serde.fromJson(inJson, Variant.class);
        final String javaOutJson = Serde.toPrettyJson(javaOutObj);
        assertJsonEquals(inJson, javaOutJson);
      }

      try (final VariantCpp inObjCo = session.variantOperations().asCpp(inObj)) {
        final Variant cppOutObj = inObjCo.asJava();
        final String cppOutJson = Serde.toPrettyJson(cppOutObj);
        Assert.assertEquals(inObj, cppOutObj);
        assertJsonEquals(inJson, cppOutJson);
        return new ObjectAndJson<>(cppOutObj, cppOutJson);
      }
    }
  }

  public static <T> ObjectAndJson<Object> testJavaBeanRoundTrip(T inObj) {
    try {
      if (inObj instanceof NativeBean) {
        throw new VeloxException("Cannot round trip NativeBean");
      }
      final Class<?> clazz = inObj.getClass();
      final ObjectMapper jsonMapper = Serde.jsonMapper();
      final String inJson = jsonMapper.writeValueAsString(inObj);
      final Object outObj = jsonMapper.readValue(inJson, clazz);
      final String outJson = jsonMapper.writeValueAsString(outObj);
      assertJsonEquals(inJson, outJson);
      return new ObjectAndJson<>(outObj, outJson);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  public static HiveColumnHandle newSampleHiveColumnHandle() {
    final Type dataType =
        ArrayType.create(
            MapType.create(
                new VarCharType(),
                new RowType(
                    List.of("id", "description"), List.of(new BigIntType(), new VarCharType()))));
    final HiveColumnHandle handle =
        new HiveColumnHandle(
            "complex_type",
            ColumnType.REGULAR,
            dataType,
            dataType,
            List.of("complex_type[1][\"foo\"].id", "complex_type[2][\"foo\"].id"));
    return handle;
  }

  public static HiveConnectorSplit newSampleHiveSplit() {
    return new HiveConnectorSplit(
        "connector-1",
        5,
        true,
        "path/to/file",
        FileFormat.ORC,
        1,
        100,
        Map.of("key", Optional.of("value")),
        OptionalInt.of(1),
        Optional.of(
            new HiveBucketConversion(
                1,
                1,
                List.of(
                    new HiveColumnHandle(
                        "t",
                        ColumnType.REGULAR,
                        new IntegerType(),
                        new IntegerType(),
                        Collections.emptyList())))),
        Map.of("sk", "sv"),
        Optional.of("extra"),
        Map.of("serde_key", "serde_value"),
        Map.of("info_key", "info_value"),
        Optional.of(new FileProperties(OptionalLong.of(100), OptionalLong.of(50))),
        Optional.of(new RowIdProperties(5, 10, "UUID-100")));
  }

  public static HiveConnectorSplit newSampleHiveSplitWithMissingFields() {
    return new HiveConnectorSplit(
        "connector-1",
        5,
        true,
        "path/to/file",
        FileFormat.ORC,
        1,
        100,
        Map.of("key", Optional.of("value")),
        OptionalInt.of(1),
        Optional.of(
            new HiveBucketConversion(
                1,
                1,
                List.of(
                    new HiveColumnHandle(
                        "t",
                        ColumnType.REGULAR,
                        new IntegerType(),
                        new IntegerType(),
                        Collections.emptyList())))),
        Map.of("sk", "sv"),
        Optional.empty(),
        Map.of("serde_key", "serde_value"),
        Map.of("info_key", "info_value"),
        Optional.of(new FileProperties(OptionalLong.empty(), OptionalLong.of(50))),
        Optional.of(new RowIdProperties(5, 10, "UUID-100")));
  }

  public static ConnectorTableHandle newSampleHiveTableHandle(RowType outputType) {
    final ConnectorTableHandle handle =
        new HiveTableHandle(
            "connector-1",
            "tab-1",
            true,
            List.of(new SubfieldFilter("complex_type[1].id", new AlwaysTrue())),
            new CallTypedExpr(new BooleanType(), Collections.emptyList(), "always_true"),
            outputType,
            Map.of("tk", "tv"));
    return handle;
  }

  public static LocationHandle newSampleLocationHandle() {
    return new LocationHandle(
        "/tmp/target-path",
        "/tmp/write-path",
        LocationHandle.TableType.EXISTING,
        "target-file-name");
  }

  public static HiveBucketProperty newSampleHiveBucketProperty() {
    return new HiveBucketProperty(
        HiveBucketProperty.Kind.PRESTO_NATIVE,
        10,
        List.of("foo", "bar"),
        List.of(new IntegerType(), new VarCharType()),
        List.of(
            new HiveSortingColumn("foo", new SortOrder(true, true)),
            new HiveSortingColumn("bar", new SortOrder(false, false))));
  }

  public static HiveInsertTableHandle newSampleHiveInsertTableHandle() {
    return new HiveInsertTableHandle(
        List.of(newSampleHiveColumnHandle()),
        newSampleLocationHandle(),
        FileFormat.PARQUET,
        newSampleHiveBucketProperty(),
        CompressionKind.ZLIB,
        Map.of("serde_key", "serde_value"),
        false,
        new HiveInsertFileNameGenerator());
  }

  public static Aggregate newSampleAggregate() {
    final Aggregate aggregate =
        new Aggregate(
            new CallTypedExpr(
                new IntegerType(),
                Collections.singletonList(FieldAccessTypedExpr.create(new IntegerType(), "foo")),
                "sum"),
            List.of(new IntegerType()),
            FieldAccessTypedExpr.create(new IntegerType(), "foo"),
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo")),
            List.of(new SortOrder(true, true)),
            true);
    return aggregate;
  }

  public static PlanNode newSampleTableScanNode(String planNodeId, RowType outputType) {
    final ConnectorTableHandle handle = SerdeTests.newSampleHiveTableHandle(outputType);
    final PlanNode scan =
        new TableScanNode(planNodeId, outputType, handle, Collections.emptyList());
    return scan;
  }

  public static AggregationNode newSampleAggregationNode(String aggNodeId, String scanNodeId) {
    final PlanNode scan =
        SerdeTests.newSampleTableScanNode(scanNodeId, SerdeTests.newSampleOutputType());
    final Aggregate aggregate = SerdeTests.newSampleAggregate();
    final AggregationNode aggregationNode =
        new AggregationNode(
            aggNodeId,
            AggregateStep.PARTIAL,
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo")),
            List.of(FieldAccessTypedExpr.create(new IntegerType(), "foo")),
            List.of("sum"),
            List.of(aggregate),
            true,
            List.of(scan),
            FieldAccessTypedExpr.create(new IntegerType(), "foo"),
            List.of(0));
    return aggregationNode;
  }

  public static RowType newSampleOutputType() {
    return new RowType(List.of("foo", "bar"), List.of(new IntegerType(), new IntegerType()));
  }

  public static class ObjectAndJson<T> {
    private final T obj;
    private final String json;

    private ObjectAndJson(T obj, String json) {
      this.obj = obj;
      this.json = json;
    }

    public T getObj() {
      return obj;
    }

    public String getJson() {
      return json;
    }
  }
}
