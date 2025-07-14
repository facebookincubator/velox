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
package com.meta.velox4j.variant;

import java.util.Base64;
import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class VarBinaryValue extends Variant {
  private final String base64;

  @JsonCreator
  public VarBinaryValue(@JsonProperty("value") String base64) {
    this.base64 = base64;
  }

  public static VarBinaryValue create(byte[] bytes) {
    return new VarBinaryValue(Base64.getEncoder().encodeToString(bytes));
  }

  @JsonGetter("value")
  public String getBase64() {
    return base64;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    VarBinaryValue that = (VarBinaryValue) o;
    return Objects.equals(base64, that.base64);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(base64);
  }

  @Override
  public String toString() {
    return "VarBinaryValue{" + "base64='" + base64 + '\'' + '}';
  }
}
