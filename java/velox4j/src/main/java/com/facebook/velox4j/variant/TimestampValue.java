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

import java.util.Objects;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;

public class TimestampValue extends Variant {
  private static final int DUMMY_VALUE = -1;
  private final Integer value;
  private final long seconds;
  private final long nanos;

  @JsonCreator
  private TimestampValue(
      @JsonProperty("value") Integer value,
      @JsonProperty("seconds") long seconds,
      @JsonProperty("nanos") long nanos) {
    // For non-null case, JSON field "value" is always -1, according to Velox's design.
    Preconditions.checkArgument(
        value == null || value == DUMMY_VALUE,
        "JSON field \"value\" has to be null or -1 in timestamp variant");
    this.value = value;
    this.seconds = seconds;
    this.nanos = nanos;
  }

  public static TimestampValue create(long seconds, long nanos) {
    return new TimestampValue(DUMMY_VALUE, seconds, nanos);
  }

  public static TimestampValue createNull() {
    return new TimestampValue(null, 0, 0);
  }

  @JsonGetter("value")
  @JsonInclude(JsonInclude.Include.ALWAYS)
  public Integer getValue() {
    return value;
  }

  @JsonGetter("seconds")
  public long getSeconds() {
    return seconds;
  }

  @JsonGetter("nanos")
  public long getNanos() {
    return nanos;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    TimestampValue that = (TimestampValue) o;
    return value == that.value && seconds == that.seconds && nanos == that.nanos;
  }

  @Override
  public int hashCode() {
    return Objects.hash(value, seconds, nanos);
  }

  @Override
  public String toString() {
    return "TimestampValue{" + "value=" + value + ", seconds=" + seconds + ", nanos=" + nanos + '}';
  }
}
