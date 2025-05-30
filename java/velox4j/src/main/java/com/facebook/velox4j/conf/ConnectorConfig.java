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
package com.facebook.velox4j.conf;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonProperty;

import com.facebook.velox4j.serializable.ISerializable;

public class ConnectorConfig extends ISerializable {
  private static final ConnectorConfig EMPTY = new ConnectorConfig(List.of());

  public static ConnectorConfig empty() {
    return EMPTY;
  }

  private final List<ConnectorConfig.Entry> values;

  @JsonCreator
  private ConnectorConfig(@JsonProperty("values") List<ConnectorConfig.Entry> values) {
    this.values = values;
  }

  public static ConnectorConfig create(Map<String, Config> values) {
    return new ConnectorConfig(
        values.entrySet().stream()
            .map(e -> new ConnectorConfig.Entry(e.getKey(), e.getValue()))
            .collect(Collectors.toList()));
  }

  @JsonGetter("values")
  public List<ConnectorConfig.Entry> values() {
    return values;
  }

  public static class Entry {
    private final String connectorId;
    private final Config config;

    @JsonCreator
    public Entry(
        @JsonProperty("connectorId") String connectorId, @JsonProperty("config") Config config) {
      this.connectorId = connectorId;
      this.config = config;
    }

    @JsonGetter("connectorId")
    public String connectorId() {
      return connectorId;
    }

    @JsonGetter("config")
    public Config config() {
      return config;
    }
  }
}
