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

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.JacksonException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.TreeNode;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.DeserializationConfig;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializer;
import com.fasterxml.jackson.databind.deser.BeanDeserializerModifier;
import com.fasterxml.jackson.databind.deser.BeanDeserializerUnsafe;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.base.Preconditions;

import com.facebook.velox4j.collection.Streams;
import com.facebook.velox4j.exception.VeloxException;

public class PolymorphicDeserializer {
  private static class AbstractDeserializer extends JsonDeserializer<Object> {
    private final Class<? extends NativeBean> baseClass;

    private AbstractDeserializer(Class<? extends NativeBean> baseClass) {
      this.baseClass = baseClass;
    }

    private SerdeRegistry findRegistry(SerdeRegistryFactory rf, ObjectNode obj) {
      final Set<String> keys = rf.keys();
      final List<String> keysInObj =
          Streams.fromIterator(obj.fieldNames())
              .filter(keys::contains)
              .collect(Collectors.toList());
      if (keysInObj.isEmpty()) {
        throw new UnsupportedOperationException("Required keys not found in JSON: " + obj);
      }
      if (keysInObj.size() > 1) {
        throw new UnsupportedOperationException("Ambiguous key annotations in JSON: " + obj);
      }
      final SerdeRegistry registry = rf.key(keysInObj.get(0));
      return registry;
    }

    private Object deserializeWithRegistry(
        JsonParser p, SerdeRegistry registry, ObjectNode objectNode) {
      final String key = registry.key();
      final String value = objectNode.get(key).asText();
      Preconditions.checkArgument(
          registry.contains(value),
          "Value %s not registered in registry: %s",
          value,
          registry.prefixAndKey());
      if (registry.isFactory(value)) {
        final SerdeRegistryFactory rf = registry.getFactory(value);
        final SerdeRegistry nextRegistry = findRegistry(rf, objectNode);
        return deserializeWithRegistry(p, nextRegistry, objectNode);
      }
      if (registry.isClass(value)) {
        final Class<?> clazz = registry.getClass(value);
        try {
          return p.getCodec().treeToValue(objectNode, clazz);
        } catch (JsonProcessingException e) {
          throw new VeloxException(e);
        }
      }
      throw new IllegalStateException();
    }

    @Override
    public Object deserialize(JsonParser p, DeserializationContext ctxt)
        throws IOException, JacksonException {
      final TreeNode treeNode = p.readValueAsTree();
      if (!treeNode.isObject()) {
        throw new UnsupportedOperationException("Not a JSON object: " + treeNode);
      }
      final ObjectNode objNode = (ObjectNode) treeNode;
      final SerdeRegistry registry =
          findRegistry(SerdeRegistryFactory.getForBaseClass(baseClass), objNode);
      return deserializeWithRegistry(p, registry, objNode);
    }
  }

  public static class Modifier extends BeanDeserializerModifier {
    private final Set<Class<? extends NativeBean>> baseClasses = new HashSet<>();

    public Modifier() {}

    @Override
    public synchronized JsonDeserializer<?> modifyDeserializer(
        DeserializationConfig config, BeanDescription beanDesc, JsonDeserializer<?> deserializer) {
      final Class<?> beanClass = beanDesc.getBeanClass();
      for (Class<? extends NativeBean> baseClass : baseClasses) {
        if (!baseClass.isAssignableFrom(beanClass)) {
          continue;
        }
        if (java.lang.reflect.Modifier.isAbstract(beanClass.getModifiers())) {
          // We use the custom deserializer for abstract classes to find the concrete type
          // information of the object.
          return new AbstractDeserializer(baseClass);
        } else {
          // Use a bean deserializer that ignores the JSON field names that are used by
          // AbstractDeserializer to find concrete type.
          Preconditions.checkState(deserializer instanceof BeanDeserializer);
          final BeanDeserializer bd = (BeanDeserializer) deserializer;
          Preconditions.checkState(BeanDeserializerUnsafe.getIgnorableProps(bd) == null);
          Preconditions.checkState(BeanDeserializerUnsafe.getIncludableProps(bd) == null);
          return bd.withByNameInclusion(
              SerdeRegistry.findKvPairs(beanClass).stream()
                  .map(SerdeRegistry.KvPair::getKey)
                  .collect(Collectors.toUnmodifiableSet()),
              null);
        }
      }
      return deserializer;
    }

    public synchronized void registerBaseClass(Class<? extends NativeBean> clazz) {
      Preconditions.checkArgument(
          !java.lang.reflect.Modifier.isInterface(clazz.getModifiers()),
          String.format(
              "Class %s is an interface which is not currently supported by PolymorphicDeserializer",
              clazz));
      Preconditions.checkArgument(
          !baseClasses.contains(clazz), "Base class already registered: %s", clazz);
      baseClasses.add(clazz);
    }
  }
}
