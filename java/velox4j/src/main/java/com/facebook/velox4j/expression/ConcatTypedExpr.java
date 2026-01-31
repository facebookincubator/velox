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

import java.util.List;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;

public class ConcatTypedExpr extends TypedExpr {
  @JsonCreator
  private ConcatTypedExpr(
      @JsonProperty("type") Type returnType, @JsonProperty("inputs") List<TypedExpr> inputs) {
    super(returnType, inputs);
    Preconditions.checkArgument(
        returnType instanceof RowType, "ConcatTypedExpr returnType should be RowType");
  }

  public static ConcatTypedExpr create(List<String> names, List<TypedExpr> inputs) {
    Preconditions.checkArgument(
        names.size() == inputs.size(),
        "ConcatTypedExpr should have same number of names and inputs");
    return new ConcatTypedExpr(
        new RowType(
            names, inputs.stream().map(TypedExpr::getReturnType).collect(Collectors.toList())),
        inputs);
  }
}
