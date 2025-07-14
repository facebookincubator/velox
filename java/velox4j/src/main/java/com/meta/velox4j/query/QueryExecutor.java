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
package com.meta.velox4j.query;

import com.meta.velox4j.jni.CppObject;
import com.meta.velox4j.jni.JniApi;

public class QueryExecutor implements CppObject {
  private final JniApi jniApi;
  private final long id;

  public QueryExecutor(JniApi jniApi, long id) {
    this.jniApi = jniApi;
    this.id = id;
  }

  @Override
  public long id() {
    return id;
  }

  public SerialTask execute() {
    return jniApi.queryExecutorExecute(this);
  }
}
