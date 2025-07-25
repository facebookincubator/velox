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
package com.facebook.velox4j.connector;

import com.fasterxml.jackson.annotation.JsonValue;

public enum FileFormat {
  UNKNOWN("unknown"),
  DWRF("dwrf"),
  RC("rc"),
  RC_TEXT("rc:text"),
  RC_BINARY("rc:binary"),
  TEXT("text"),
  JSON("json"),
  PARQUET("parquet"),
  NIMBLE("nimble"),
  ORC("orc"),
  SST("sst");

  private final String value;

  FileFormat(String value) {
    this.value = value;
  }

  @JsonValue
  public String toValue() {
    return value;
  }
}
