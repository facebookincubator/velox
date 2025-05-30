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
package com.facebook.velox4j.query;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.serde.Serde;

public class SerialTaskStats {
  private final ArrayNode planStatsDynamic;

  private SerialTaskStats(JsonNode planStatsDynamic) {
    this.planStatsDynamic = (ArrayNode) planStatsDynamic;
  }

  public static SerialTaskStats fromJson(String statsJson) {
    final JsonNode dynamic = Serde.parseTree(statsJson);
    final JsonNode planStatsDynamic =
        Preconditions.checkNotNull(
            dynamic.get("planStats"), "Plan statistics not found in task statistics");
    return new SerialTaskStats(planStatsDynamic);
  }

  public ObjectNode planStats(String planNodeId) {
    final List<ObjectNode> out = new ArrayList<>();
    for (JsonNode each : planStatsDynamic) {
      if (Objects.equals(each.get("planNodeId").asText(), planNodeId)) {
        out.add((ObjectNode) each);
      }
    }
    if (out.isEmpty()) {
      throw new VeloxException(
          String.format("Statistics for plan node not found, node ID: %s", planNodeId));
    }
    if (out.size() != 1) {
      throw new VeloxException(
          String.format(
              "More than one nodes (%d total) with the same node ID found in task statistics, node ID: %s",
              out.size(), planNodeId));
    }
    return out.get(0);
  }
}
