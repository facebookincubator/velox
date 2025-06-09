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
package com.facebook.velox4j.iterator;

import com.facebook.velox4j.jni.CalledFromNative;

/**
 * An ExternalStream that is backed by a down-iterator. What is down-iterator: A down-iterator is an
 * iterator passed From Java to C++ for Velox to read data from Java.
 */
public interface DownIterator {
  enum State {
    AVAILABLE(0),
    BLOCKED(1),
    FINISHED(2);

    private final int id;

    State(int id) {
      this.id = id;
    }

    public int getId() {
      return id;
    }
  }

  /** Gets the next state. */
  @CalledFromNative
  int advance();

  /**
   * Called once `advance` returns `BLOCKED` state to wait until the state gets refreshed, either by
   * the next row-vector is ready for reading or by end of stream.
   */
  @CalledFromNative
  void waitFor() throws InterruptedException;

  /** Called to close the iterator. */
  @CalledFromNative
  long get();

  /** Closes the down-iterator. */
  @CalledFromNative
  void close();
}
