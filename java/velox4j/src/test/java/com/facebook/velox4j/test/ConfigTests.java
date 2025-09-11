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
package com.facebook.velox4j.test;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.conf.ConnectorConfig;

public final class ConfigTests {
  private ConfigTests() {}

  public static Config randomConfig() {
    final Map<String, String> entries = new HashMap<>();
    for (int i = 0; i < 100; i++) {
      entries.put(UUID.randomUUID().toString(), UUID.randomUUID().toString());
    }
    final Config config = Config.create(entries);
    return config;
  }

  public static ConnectorConfig randomConnectorConfig() {
    final Map<String, Config> values = new HashMap<>();
    for (int i = 0; i < 10; i++) {
      values.put(UUID.randomUUID().toString(), randomConfig());
    }
    return ConnectorConfig.create(values);
  }
}
