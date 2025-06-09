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
package com.fasterxml.jackson.databind.deser;

import java.util.Set;

/** Package-hacking the protected APIs of BeanDeserializer for Velox4J's use. */
public final class BeanDeserializerUnsafe {
  public static Set<String> getIgnorableProps(BeanDeserializer beanDeserializer) {
    return beanDeserializer._ignorableProps;
  }

  public static Set<String> getIncludableProps(BeanDeserializer beanDeserializer) {
    return beanDeserializer._includableProps;
  }
}
