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
package com.facebook.velox4j.resource;

import java.io.File;

public class PlainResourceFile implements ResourceFile {
  private final String container;
  private final String name;

  PlainResourceFile(String container, String name) {
    this.container = container;
    this.name = name;
  }

  @Override
  public String container() {
    return container;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public void copyTo(File file) {
    Resources.copyResource(container + File.separator + name, file);
  }
}
