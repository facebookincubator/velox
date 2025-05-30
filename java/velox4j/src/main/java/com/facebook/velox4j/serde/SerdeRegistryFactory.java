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
package com.facebook.velox4j.serde;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import com.facebook.velox4j.exception.VeloxException;

public class SerdeRegistryFactory {
  private static final Map<Class<? extends NativeBean>, SerdeRegistryFactory> INSTANCES =
      new ConcurrentHashMap<>();

  public static SerdeRegistryFactory createForBaseClass(Class<? extends NativeBean> clazz) {
    return INSTANCES.compute(
        clazz,
        (k, v) -> {
          if (v != null) {
            throw new VeloxException("SerdeRegistryFactory already exists for " + k);
          }
          checkClass(k, INSTANCES.keySet());
          return new SerdeRegistryFactory(Collections.emptyList());
        });
  }

  private static void checkClass(
      Class<? extends NativeBean> clazz, Set<Class<? extends NativeBean>> others) {
    // The class to register should not be assignable from / to each other.
    for (Class<? extends NativeBean> other : others) {
      if (clazz.isAssignableFrom(other)) {
        throw new VeloxException(
            String.format(
                "Class %s is not register-able because it is assignable from %s", clazz, other));
      }
      if (other.isAssignableFrom(clazz)) {
        throw new VeloxException(
            String.format(
                "Class %s is not register-able because it is assignable to %s", clazz, other));
      }
    }
  }

  public static SerdeRegistryFactory getForBaseClass(Class<? extends NativeBean> clazz) {
    return INSTANCES.get(clazz);
  }

  private final Map<String, SerdeRegistry> registries = new HashMap<>();
  private final List<SerdeRegistry.KvPair> kvs;

  SerdeRegistryFactory(List<SerdeRegistry.KvPair> kvs) {
    this.kvs = kvs;
  }

  public SerdeRegistry key(String key) {
    synchronized (this) {
      if (!registries.containsKey(key)) {
        registries.put(key, new SerdeRegistry(kvs, key));
      }
      return registries.get(key);
    }
  }

  public Set<String> keys() {
    synchronized (this) {
      return registries.keySet();
    }
  }
}
