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
package com.facebook.velox4j.variant;

import com.facebook.velox4j.serde.Serde;
import com.facebook.velox4j.serde.SerdeRegistry;
import com.facebook.velox4j.serde.SerdeRegistryFactory;

public class VariantRegistry {
  private static final SerdeRegistry TYPE_REGISTRY =
      SerdeRegistryFactory.createForBaseClass(Variant.class).key("type");

  private VariantRegistry() {}

  public static void registerAll() {
    Serde.registerBaseClass(Variant.class);
    TYPE_REGISTRY.registerClass("BOOLEAN", BooleanValue.class);
    TYPE_REGISTRY.registerClass("TINYINT", TinyIntValue.class);
    TYPE_REGISTRY.registerClass("SMALLINT", SmallIntValue.class);
    TYPE_REGISTRY.registerClass("INTEGER", IntegerValue.class);
    TYPE_REGISTRY.registerClass("BIGINT", BigIntValue.class);
    TYPE_REGISTRY.registerClass("HUGEINT", HugeIntValue.class);
    TYPE_REGISTRY.registerClass("REAL", RealValue.class);
    TYPE_REGISTRY.registerClass("DOUBLE", DoubleValue.class);
    TYPE_REGISTRY.registerClass("VARCHAR", VarCharValue.class);
    TYPE_REGISTRY.registerClass("VARBINARY", VarBinaryValue.class);
    TYPE_REGISTRY.registerClass("TIMESTAMP", TimestampValue.class);
    TYPE_REGISTRY.registerClass("ARRAY", ArrayValue.class);
    TYPE_REGISTRY.registerClass("MAP", MapValue.class);
    TYPE_REGISTRY.registerClass("ROW", RowValue.class);
  }
}
