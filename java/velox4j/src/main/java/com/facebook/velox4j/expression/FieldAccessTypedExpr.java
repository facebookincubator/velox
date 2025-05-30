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
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;

public class FieldAccessTypedExpr extends TypedExpr {
  private final String fieldName;

  @JsonCreator
  private FieldAccessTypedExpr(
      @JsonProperty("type") Type returnType,
      @JsonProperty("inputs") List<TypedExpr> inputs,
      @JsonProperty("fieldName") String fieldName) {
    super(returnType, inputs);
    this.fieldName = fieldName;
    Preconditions.checkArgument(
        getInputs().size() <= 1,
        "FieldAccessTypedExpr should have 0 or 1 input, but has %s",
        getInputs().size());
  }

  public static FieldAccessTypedExpr create(Type returnType, String fieldName) {
    return new FieldAccessTypedExpr(returnType, Collections.emptyList(), fieldName);
  }

  public static FieldAccessTypedExpr create(TypedExpr input, String fieldName) {
    final Type inputType = input.getReturnType();
    Preconditions.checkArgument(
        inputType instanceof RowType, "FieldAccessTypedExpr input should be RowType");
    final Type returnType = ((RowType) inputType).findChild(fieldName);
    return new FieldAccessTypedExpr(returnType, Collections.singletonList(input), fieldName);
  }

  @JsonGetter("fieldName")
  public String getFieldName() {
    return fieldName;
  }
}
