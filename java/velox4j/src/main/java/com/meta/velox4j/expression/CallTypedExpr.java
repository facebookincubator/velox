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
package com.meta.velox4j.expression;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.meta.velox4j.type.Type;

public class CallTypedExpr extends TypedExpr {
  private final String functionName;

  @JsonCreator
  public CallTypedExpr(
      @JsonProperty("type") Type returnType,
      @JsonProperty("inputs") List<TypedExpr> inputs,
      @JsonProperty("functionName") String functionName) {
    super(returnType, inputs);
    this.functionName = functionName;
  }

  @JsonGetter("functionName")
  public String getFunctionName() {
    return functionName;
  }
}
