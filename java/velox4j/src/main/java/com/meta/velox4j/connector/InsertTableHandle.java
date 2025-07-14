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
package com.meta.velox4j.connector;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

public class InsertTableHandle {
  private final String connectorId;
  private final ConnectorInsertTableHandle connectorInsertTableHandle;

  @JsonCreator
  public InsertTableHandle(
      @JsonProperty("connectorId") String connectorId,
      @JsonProperty("connectorInsertTableHandle")
          ConnectorInsertTableHandle connectorInsertTableHandle) {
    this.connectorId = connectorId;
    this.connectorInsertTableHandle = connectorInsertTableHandle;
  }

  @JsonGetter("connectorId")
  public String getConnectorId() {
    return connectorId;
  }

  @JsonGetter("connectorInsertTableHandle")
  public ConnectorInsertTableHandle connectorInsertTableHandle() {
    return connectorInsertTableHandle;
  }
}
