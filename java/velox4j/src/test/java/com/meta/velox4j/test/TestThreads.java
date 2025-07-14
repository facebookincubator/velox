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
package com.meta.velox4j.test;

import com.google.common.util.concurrent.UncaughtExceptionHandlers;

public final class TestThreads {
  public static Thread newTestThread(Runnable target) {
    final Thread t = new Thread(target);
    t.setUncaughtExceptionHandler(UncaughtExceptionHandlers.systemExit());
    return t;
  }
}
