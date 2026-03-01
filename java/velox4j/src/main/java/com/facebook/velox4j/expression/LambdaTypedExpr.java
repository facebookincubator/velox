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

import com.facebook.velox4j.type.FunctionType;
import com.facebook.velox4j.type.RowType;
import com.facebook.velox4j.type.Type;

public class LambdaTypedExpr extends TypedExpr {
  private final Type signature;
  private final TypedExpr body;

  @JsonCreator
  private LambdaTypedExpr(
      @JsonProperty("type") Type returnType,
      @JsonProperty("signature") Type signature,
      @JsonProperty("body") TypedExpr body) {
    super(returnType, Collections.emptyList());
    this.signature = signature;
    this.body = body;
    Preconditions.checkArgument(
        returnType instanceof FunctionType, "LambdaTypedExpr returnType should be FunctionType");
  }

  public static LambdaTypedExpr create(RowType signature, TypedExpr body) {
    final FunctionType returnType =
        FunctionType.create(signature.getChildren(), body.getReturnType());
    return new LambdaTypedExpr(returnType, signature, body);
  }

  @JsonGetter("signature")
  public Type getSignature() {
    return signature;
  }

  @JsonGetter("body")
  public TypedExpr getBody() {
    return body;
  }
}
