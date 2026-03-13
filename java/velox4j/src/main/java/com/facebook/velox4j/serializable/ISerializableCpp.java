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
package com.facebook.velox4j.serializable;

import com.facebook.velox4j.jni.CppObject;
import com.facebook.velox4j.jni.StaticJniApi;

/** Binds a CPP ISerializable object. */
public class ISerializableCpp implements CppObject {
  private final long id;

  public ISerializableCpp(long id) {
    this.id = id;
  }

  @Override
  public long id() {
    return id;
  }

  public ISerializable asJava() {
    return StaticJniApi.get().iSerializableAsJava(this);
  }
}
