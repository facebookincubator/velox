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

import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import com.google.common.base.Preconditions;

import com.facebook.velox4j.exception.VeloxException;

public class SerdeRegistry {
  private static final Map<Class<? extends NativeBean>, List<KvPair>> CLASS_TO_KVS =
      new ConcurrentHashMap<>();

  private final Map<String, SerdeRegistryFactory> subFactories = new HashMap<>();
  private final Map<String, Class<? extends NativeBean>> classes = new HashMap<>();
  private final List<KvPair> kvs;
  private final String key;
  private final String prefixAndKey;

  public SerdeRegistry(List<KvPair> kvs, String key) {
    this.kvs = Collections.unmodifiableList(kvs);
    this.key = key;
    this.prefixAndKey = String.format("%s.%s", kvs.toString(), key);
  }

  public String key() {
    return key;
  }

  public String prefixAndKey() {
    return prefixAndKey;
  }

  private List<KvPair> withNewKv(KvPair kv) {
    final List<KvPair> out = new ArrayList<>(kvs);
    out.add(kv);
    return out;
  }

  public SerdeRegistryFactory registerFactory(String value) {
    synchronized (this) {
      checkDup(value);
      final SerdeRegistryFactory factory =
          new SerdeRegistryFactory(withNewKv(new KvPair(key, value)));
      subFactories.put(value, factory);
      return factory;
    }
  }

  public void registerClass(String value, Class<? extends NativeBean> clazz) {
    Preconditions.checkArgument(
        !Modifier.isAbstract(clazz.getModifiers()), "Cannot register abstract class: " + clazz);
    synchronized (this) {
      checkDup(value);
      classes.put(value, clazz);
    }
    if (CLASS_TO_KVS.put(clazz, withNewKv(new KvPair(key, value))) != null) {
      throw new VeloxException("Class SerDe already registered: " + clazz);
    }
  }

  public boolean contains(String value) {
    synchronized (this) {
      return isFactory(value) || isClass(value);
    }
  }

  public boolean isFactory(String value) {
    synchronized (this) {
      return subFactories.containsKey(value);
    }
  }

  public boolean isClass(String value) {
    synchronized (this) {
      return classes.containsKey(value);
    }
  }

  public SerdeRegistryFactory getFactory(String value) {
    synchronized (this) {
      if (!subFactories.containsKey(value)) {
        throw new VeloxException(
            String.format("Value %s.%s is not added as a sub-factory: ", prefixAndKey, value));
      }
      return subFactories.get(value);
    }
  }

  public Class<?> getClass(String value) {
    synchronized (this) {
      if (!classes.containsKey(value)) {
        throw new VeloxException(
            String.format("Value %s.%s is not added as a class: ", prefixAndKey, value));
      }
      return classes.get(value);
    }
  }

  public static boolean isRegistered(Class<?> clazz) {
    return CLASS_TO_KVS.containsKey(clazz);
  }

  public static List<KvPair> findKvPairs(Class<?> clazz) {
    if (!isRegistered(clazz)) {
      throw new VeloxException("Class SerDe not registered: " + clazz);
    }
    return CLASS_TO_KVS.get(clazz);
  }

  private void checkDup(String value) {
    if (subFactories.containsKey(value)) {
      throw new VeloxException(
          String.format("Value %s.%s already added as a sub-factory: ", prefixAndKey, value));
    }
    if (classes.containsKey(value)) {
      throw new VeloxException(
          String.format("Value %s.%s already added as a class: ", prefixAndKey, value));
    }
  }

  public static class KvPair {
    private final String key;
    private final String value;

    public KvPair(String key, String value) {
      this.key = key;
      this.value = value;
    }

    public String getKey() {
      return key;
    }

    public String getValue() {
      return value;
    }
  }
}
