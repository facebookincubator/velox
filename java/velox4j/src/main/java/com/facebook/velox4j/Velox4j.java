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
package com.facebook.velox4j;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import com.google.common.base.Preconditions;

import com.facebook.velox4j.conf.Config;
import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.jni.JniLibLoader;
import com.facebook.velox4j.jni.JniWorkspace;
import com.facebook.velox4j.jni.StaticJniApi;
import com.facebook.velox4j.memory.MemoryManager;
import com.facebook.velox4j.serializable.ISerializableRegistry;
import com.facebook.velox4j.session.Session;
import com.facebook.velox4j.variant.VariantRegistry;

public class Velox4j {
  private static final AtomicBoolean initialized = new AtomicBoolean(false);
  private static final Map<String, String> globalConfMap = new LinkedHashMap<>();

  /**
   * Configure Velox4J with a key-value pair. The available configuration entries are defined in the
   * C++ source file `velox4j/init/Config.cpp`.
   *
   * <p>The method should only be called before `initialize` is called. Exception is thrown
   * otherwise.
   */
  public static void configure(String key, String value) {
    Preconditions.checkNotNull(key, "Key cannot be null");
    Preconditions.checkNotNull(value, "Value of key %s cannot be null", key);
    synchronized (globalConfMap) {
      if (globalConfMap.containsKey(key)) {
        final String oldValue = Preconditions.checkNotNull(globalConfMap.get(key));
        if (Objects.equals(oldValue, value)) {
          // The configure call actually does nothing. Ignore.
          return;
        }
      }
      // If Velox4J was already initialized, throw.
      if (initialized.get()) {
        throw new VeloxException(
            "Could not change configuration after Velox4J was already initialized");
      }
      // Apply the configuration change.
      globalConfMap.put(key, value);
    }
  }

  /**
   * Initializes Verlox4J. Should be called once and only once before any Velox4J functionalities
   * are actually invoked.
   */
  public static void initialize() {
    if (!initialized.compareAndSet(false, true)) {
      throw new VeloxException("Velox4J has already been initialized");
    }
    initializeInternal();
  }

  /** Creates a new Velox4J session. */
  public static Session newSession(MemoryManager memoryManager) {
    return StaticJniApi.get().createSession(memoryManager);
  }

  private static void initializeInternal() {
    synchronized (globalConfMap) {
      final Config globalConf = Config.create(globalConfMap);
      JniLibLoader.loadAll(JniWorkspace.getDefault().getSubDir("lib"));
      VariantRegistry.registerAll();
      ISerializableRegistry.registerAll();
      StaticJniApi.get().initialize(globalConf);
    }
  }
}
