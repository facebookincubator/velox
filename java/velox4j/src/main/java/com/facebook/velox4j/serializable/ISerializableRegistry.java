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
package com.facebook.velox4j.serializable;

import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.conf.ConnectorConfig;
import com.facebook.velox4j.connector.*;
import com.facebook.velox4j.eval.Evaluation;
import com.facebook.velox4j.expression.CallTypedExpr;
import com.facebook.velox4j.expression.CastTypedExpr;
import com.facebook.velox4j.expression.ConcatTypedExpr;
import com.facebook.velox4j.expression.ConstantTypedExpr;
import com.facebook.velox4j.expression.DereferenceTypedExpr;
import com.facebook.velox4j.expression.FieldAccessTypedExpr;
import com.facebook.velox4j.expression.InputTypedExpr;
import com.facebook.velox4j.expression.LambdaTypedExpr;
import com.facebook.velox4j.filter.AlwaysTrue;
import com.facebook.velox4j.plan.AggregationNode;
import com.facebook.velox4j.plan.FilterNode;
import com.facebook.velox4j.plan.HashJoinNode;
import com.facebook.velox4j.plan.LimitNode;
import com.facebook.velox4j.plan.OrderByNode;
import com.facebook.velox4j.plan.ProjectNode;
import com.facebook.velox4j.plan.TableScanNode;
import com.facebook.velox4j.plan.TableWriteNode;
import com.facebook.velox4j.plan.ValuesNode;
import com.facebook.velox4j.query.Query;
import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.serde.SerdeRegistry;
import com.facebook.velox4j.serde.SerdeRegistryFactory;
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

public final class ISerializableRegistry {
  private static final SerdeRegistry NAME_REGISTRY =
      SerdeRegistryFactory.createForBaseClass(ISerializable.class).key("name");

  private ISerializableRegistry() {}

  public static void registerAll() {
    Serde.registerBaseClass(ISerializable.class);
    registerTypes();
    registerExprs();
    registerConnectors();
    registerFilters();
    registerPlanNodes();
    retisterConfig();
    registerEvaluation();
    registerQuery();
  }

  private static void registerTypes() {
    final SerdeRegistry typeRegistry = NAME_REGISTRY.registerFactory("Type").key("type");
    typeRegistry.registerClass("BOOLEAN", BooleanType.class);
    typeRegistry.registerClass("TINYINT", TinyIntType.class);
    typeRegistry.registerClass("SMALLINT", SmallIntType.class);
    typeRegistry.registerClass("INTEGER", IntegerType.class);
    typeRegistry.registerClass("BIGINT", BigIntType.class);
    typeRegistry.registerClass("HUGEINT", HugeIntType.class);
    typeRegistry.registerClass("REAL", RealType.class);
    typeRegistry.registerClass("DOUBLE", DoubleType.class);
    typeRegistry.registerClass("VARCHAR", VarCharType.class);
    typeRegistry.registerClass("VARBINARY", VarbinaryType.class);
    typeRegistry.registerClass("TIMESTAMP", TimestampType.class);
    typeRegistry.registerClass("ARRAY", ArrayType.class);
    typeRegistry.registerClass("MAP", MapType.class);
    typeRegistry.registerClass("ROW", RowType.class);
    typeRegistry.registerClass("FUNCTION", FunctionType.class);
    typeRegistry.registerClass("UNKNOWN", UnknownType.class);
    typeRegistry.registerClass("OPAQUE", OpaqueType.class);
    typeRegistry.registerClass("DECIMAL", DecimalType.class);
    NAME_REGISTRY
        .registerFactory("IntervalDayTimeType")
        .key("type")
        .registerClass("INTERVAL DAY TO SECOND", IntervalDayTimeType.class);
    NAME_REGISTRY
        .registerFactory("IntervalYearMonthType")
        .key("type")
        .registerClass("INTERVAL YEAR TO MONTH", IntervalYearMonthType.class);
    NAME_REGISTRY.registerFactory("DateType").key("type").registerClass("DATE", DateType.class);
  }

  private static void registerExprs() {
    NAME_REGISTRY.registerClass("CallTypedExpr", CallTypedExpr.class);
    NAME_REGISTRY.registerClass("CastTypedExpr", CastTypedExpr.class);
    NAME_REGISTRY.registerClass("ConcatTypedExpr", ConcatTypedExpr.class);
    NAME_REGISTRY.registerClass("ConstantTypedExpr", ConstantTypedExpr.class);
    NAME_REGISTRY.registerClass("DereferenceTypedExpr", DereferenceTypedExpr.class);
    NAME_REGISTRY.registerClass("FieldAccessTypedExpr", FieldAccessTypedExpr.class);
    NAME_REGISTRY.registerClass("InputTypedExpr", InputTypedExpr.class);
    NAME_REGISTRY.registerClass("LambdaTypedExpr", LambdaTypedExpr.class);
  }

  private static void registerConnectors() {
    NAME_REGISTRY.registerClass("HiveColumnHandle", HiveColumnHandle.class);
    NAME_REGISTRY.registerClass("HiveConnectorSplit", HiveConnectorSplit.class);
    NAME_REGISTRY.registerClass("HiveTableHandle", HiveTableHandle.class);
    NAME_REGISTRY.registerClass("ExternalStreamConnectorSplit", ExternalStreamConnectorSplit.class);
    NAME_REGISTRY.registerClass("ExternalStreamTableHandle", ExternalStreamTableHandle.class);
    NAME_REGISTRY.registerClass("LocationHandle", LocationHandle.class);
    NAME_REGISTRY.registerClass("HiveSortingColumn", HiveSortingColumn.class);
    NAME_REGISTRY.registerClass("HiveBucketProperty", HiveBucketProperty.class);
    NAME_REGISTRY.registerClass("HiveInsertTableHandle", HiveInsertTableHandle.class);
    NAME_REGISTRY.registerClass("HiveInsertFileNameGenerator", HiveInsertFileNameGenerator.class);
  }

  private static void registerFilters() {
    NAME_REGISTRY.registerClass("AlwaysTrue", AlwaysTrue.class);
  }

  private static void registerPlanNodes() {
    NAME_REGISTRY.registerClass("ValuesNode", ValuesNode.class);
    NAME_REGISTRY.registerClass("TableScanNode", TableScanNode.class);
    NAME_REGISTRY.registerClass("AggregationNode", AggregationNode.class);
    NAME_REGISTRY.registerClass("ProjectNode", ProjectNode.class);
    NAME_REGISTRY.registerClass("FilterNode", FilterNode.class);
    NAME_REGISTRY.registerClass("HashJoinNode", HashJoinNode.class);
    NAME_REGISTRY.registerClass("OrderByNode", OrderByNode.class);
    NAME_REGISTRY.registerClass("LimitNode", LimitNode.class);
    NAME_REGISTRY.registerClass("TableWriteNode", TableWriteNode.class);
  }

  private static void retisterConfig() {
    NAME_REGISTRY.registerClass("velox4j.Config", Config.class);
    NAME_REGISTRY.registerClass("velox4j.ConnectorConfig", ConnectorConfig.class);
  }

  private static void registerEvaluation() {
    NAME_REGISTRY.registerClass("velox4j.Evaluation", Evaluation.class);
  }

  private static void registerQuery() {
    NAME_REGISTRY.registerClass("velox4j.Query", Query.class);
  }
}
