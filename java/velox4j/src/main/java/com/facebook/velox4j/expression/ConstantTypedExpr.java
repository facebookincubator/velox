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
package com.facebook.velox4j.expression;

import java.util.Collections;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.data.BaseVector;
import com.facebook.velox4j.data.BaseVectors;
import com.facebook.velox4j.data.VectorEncoding;
import com.facebook.velox4j.jni.StaticJniApi;
import com.facebook.velox4j.type.Type;
import com.facebook.velox4j.variant.Variant;

public class ConstantTypedExpr extends TypedExpr {
  private final Variant value;
  private final String serializedVector;

  @JsonCreator
  public ConstantTypedExpr(
      @JsonProperty("type") Type returnType,
      @JsonProperty("value") Variant value,
      @JsonProperty("valueVector") String serializedVector) {
    super(returnType, Collections.emptyList());
    Preconditions.checkArgument(
        (value == null) != (serializedVector == null),
        "Either a variant value or a serialized value vector should be provided when creating ConstantTypedExpr");
    this.value = value;
    this.serializedVector = serializedVector;
  }

  public static ConstantTypedExpr create(BaseVector vector) {
    final BaseVector constVector;
    if (vector.getEncoding() == VectorEncoding.CONSTANT) {
      constVector = vector;
    } else {
      constVector = vector.wrapInConstant(1, 0);
    }
    final String serialized = BaseVectors.serializeOne(constVector);
    final Type type = vector.getType();
    return new ConstantTypedExpr(type, null, serialized);
  }

  public static ConstantTypedExpr create(Variant value) {
    return new ConstantTypedExpr(StaticJniApi.get().variantInferType(value), value, null);
  }

  @JsonGetter("value")
  public Variant getValue() {
    return value;
  }

  @JsonGetter("valueVector")
  public String getSerializedVector() {
    return serializedVector;
  }
}
