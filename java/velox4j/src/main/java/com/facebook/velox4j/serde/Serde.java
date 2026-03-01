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

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.datatype.jdk8.Jdk8Module;

import com.facebook.velox4j.exception.VeloxException;

public final class Serde {
  private static final PolymorphicDeserializer.Modifier DESER_MOD =
      new PolymorphicDeserializer.Modifier();
  private static final PolymorphicSerializer.Modifier SER_MOD =
      new PolymorphicSerializer.Modifier();
  private static final ObjectMapper JSON = newVeloxJsonMapper(DESER_MOD, SER_MOD);

  private static ObjectMapper newVeloxJsonMapper(
      PolymorphicDeserializer.Modifier deserializerModifier,
      PolymorphicSerializer.Modifier serializerModifier) {
    final JsonMapper.Builder jsonMapper = JsonMapper.builder();
    jsonMapper.serializationInclusion(JsonInclude.Include.NON_NULL);
    jsonMapper.enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION);
    jsonMapper.enable(JsonGenerator.Feature.STRICT_DUPLICATE_DETECTION);
    jsonMapper.disable(MapperFeature.AUTO_DETECT_FIELDS);
    jsonMapper.disable(MapperFeature.AUTO_DETECT_IS_GETTERS);
    jsonMapper.disable(MapperFeature.AUTO_DETECT_GETTERS);
    jsonMapper.disable(MapperFeature.AUTO_DETECT_SETTERS);
    jsonMapper.disable(MapperFeature.AUTO_DETECT_CREATORS);
    jsonMapper.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
    jsonMapper.disable(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES);
    jsonMapper.addModule(new Jdk8Module());
    jsonMapper.addModule(new SimpleModule().setDeserializerModifier(deserializerModifier));
    jsonMapper.addModule(new SimpleModule().setSerializerModifier(serializerModifier));
    return jsonMapper.build();
  }

  public static void registerBaseClass(Class<? extends NativeBean> baseClass) {
    DESER_MOD.registerBaseClass(baseClass);
    SER_MOD.registerBaseClass(baseClass);
  }

  static ObjectMapper jsonMapper() {
    return JSON;
  }

  public static String toJson(NativeBean bean) {
    try {
      return JSON.writer().writeValueAsString(bean);
    } catch (JsonProcessingException e) {
      throw new VeloxException(e);
    }
  }

  public static String toPrettyJson(NativeBean bean) {
    try {
      return JSON.writerWithDefaultPrettyPrinter().writeValueAsString(bean);
    } catch (JsonProcessingException e) {
      throw new VeloxException(e);
    }
  }

  public static JsonNode parseTree(String json) {
    try {
      return JSON.reader().readTree(json);
    } catch (IOException e) {
      throw new VeloxException(e);
    }
  }

  public static <T extends NativeBean> T fromJson(String json, Class<? extends T> valueType) {
    try {
      return JSON.reader().readValue(json, valueType);
    } catch (IOException e) {
      throw new VeloxException(e);
    }
  }
}
