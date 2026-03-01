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
package com.facebook.velox4j.jni;

/**
 * A CppObject is a Java-side representation of a C++ side object (usually managed by a C++ smart
 * pointer).
 */
public interface CppObject extends AutoCloseable {
  long id();

  /**
   * Closes the associated C++ side object. In practice, this only releases the JNI reference of the
   * smart pointer that manages the C++ object. Hence, if there are other references alive in C++
   * code, the object will not be immediately destroyed.
   */
  @Override
  default void close() {
    StaticJniApi.get().releaseCppObject(this);
  }
}
