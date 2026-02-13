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

import com.facebook.velox4j.type.Type;

public class CastTypedExpr extends TypedExpr {
  private final boolean isTryCast;

  @JsonCreator
  private CastTypedExpr(
      @JsonProperty("type") Type returnType,
      @JsonProperty("inputs") List<TypedExpr> inputs,
      @JsonProperty("isTryCast") boolean isTryCast) {
    super(returnType, inputs);
    Preconditions.checkArgument(
        inputs.size() == 1, "CastTypedExpr should have 1 input, but has %s", inputs.size());
    this.isTryCast = isTryCast;
  }

  public static CastTypedExpr create(Type returnType, TypedExpr input, boolean isTryCast) {
    return new CastTypedExpr(returnType, Collections.singletonList(input), isTryCast);
  }

  @JsonGetter("isTryCast")
  public boolean isTryCast() {
    return isTryCast;
  }
}
