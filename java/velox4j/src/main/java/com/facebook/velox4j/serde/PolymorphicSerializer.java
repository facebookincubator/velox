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
import java.util.*;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.BeanDescription;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializationConfig;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.ser.BeanSerializer;
import com.fasterxml.jackson.databind.ser.BeanSerializerModifier;
import com.fasterxml.jackson.databind.ser.std.BeanSerializerBase;
import com.fasterxml.jackson.databind.ser.std.ToEmptyObjectSerializer;
import com.google.common.base.Preconditions;

public final class PolymorphicSerializer {
  private PolymorphicSerializer() {}

  private static class EmptyBeanSerializer extends JsonSerializer<Object> {
    @Override
    public void serialize(Object bean, JsonGenerator gen, SerializerProvider serializers)
        throws IOException {
      gen.writeStartObject();
      final Class<?> clazz = bean.getClass();
      final List<SerdeRegistry.KvPair> kvs = SerdeRegistry.findKvPairs(clazz);
      for (SerdeRegistry.KvPair kv : kvs) {
        gen.writeStringField(kv.getKey(), kv.getValue());
      }
      gen.writeEndObject();
    }
  }

  private static final class NonEmptyBeanSerializer extends BeanSerializer {
    public NonEmptyBeanSerializer(BeanSerializerBase base) {
      super(base);
    }

    @Override
    protected void serializeFields(Object bean, JsonGenerator gen, SerializerProvider provider)
        throws IOException {
      final Class<?> clazz = bean.getClass();
      final List<SerdeRegistry.KvPair> kvs = SerdeRegistry.findKvPairs(clazz);
      for (SerdeRegistry.KvPair kv : kvs) {
        gen.writeStringField(kv.getKey(), kv.getValue());
      }
      super.serializeFields(bean, gen, provider);
    }
  }

  public static class Modifier extends BeanSerializerModifier {
    private final Set<Class<? extends NativeBean>> baseClasses = new HashSet<>();

    public Modifier() {}

    @Override
    public synchronized JsonSerializer<?> modifySerializer(
        SerializationConfig config, BeanDescription beanDesc, JsonSerializer<?> serializer) {
      for (Class<? extends NativeBean> baseClass : baseClasses) {
        if (!baseClass.isAssignableFrom(beanDesc.getBeanClass())) {
          continue;
        }
        if (serializer instanceof ToEmptyObjectSerializer) {
          return new EmptyBeanSerializer();
        }
        if (serializer instanceof BeanSerializerBase) {
          return new NonEmptyBeanSerializer(((BeanSerializerBase) serializer));
        }
      }
      return serializer;
    }

    public synchronized void registerBaseClass(Class<? extends NativeBean> clazz) {
      Preconditions.checkArgument(
          !java.lang.reflect.Modifier.isInterface(clazz.getModifiers()),
          String.format(
              "Class %s is an interface which is not currently supported by PolymorphicSerializer",
              clazz));
      Preconditions.checkArgument(
          !baseClasses.contains(clazz), "Base class already registered: %s", clazz);
      baseClasses.add(clazz);
    }
  }
}
