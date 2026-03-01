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

import java.io.File;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.google.common.base.Preconditions;
import com.google.common.base.StandardSystemProperty;

import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.resource.ResourceFile;
import com.facebook.velox4j.resource.Resources;

public class JniLibLoader {
  private static final AtomicBoolean LOADED = new AtomicBoolean(false);

  private static final String LIB_CONTAINER =
      String.format(
          "velox4j-lib/%s/%s",
          StandardSystemProperty.OS_NAME.value(), StandardSystemProperty.OS_ARCH.value());
  private static final Pattern LIB_PATTERN = Pattern.compile("^.+$");
  private static final String VELOX4J_LIB_NAME = "libvelox4j.so";

  public static void loadAll(File workDir) {
    if (!LOADED.compareAndSet(false, true)) {
      throw new VeloxException("Libraries were already loaded");
    }
    Preconditions.checkArgument(
        workDir.isDirectory(), "Work directory %s is not a directory", workDir);
    final List<ResourceFile> libFiles = Resources.getResources(LIB_CONTAINER, LIB_PATTERN);
    if (libFiles.isEmpty()) {
      throw new VeloxException(
          String.format(
              "Library container %s not found in classpath. Please check whether the current platform is supported by the Jar",
              LIB_CONTAINER));
    }
    final List<ResourceFile> velox4jLibFiles =
        libFiles.stream()
            .filter(f -> f.name().equals(VELOX4J_LIB_NAME))
            .collect(Collectors.toList());
    Preconditions.checkArgument(velox4jLibFiles.size() == 1, "Velox4J library not found");
    System.out.printf("Found required libraries in container %s.%n", LIB_CONTAINER);
    final ResourceFile velox4jLibFile = velox4jLibFiles.get(0);
    for (ResourceFile libFile : libFiles) {
      final File copied = workDir.toPath().resolve(libFile.name()).toFile();
      System.out.printf("Copying library %s/%s to %s...%n", LIB_CONTAINER, libFile.name(), copied);
      libFile.copyTo(copied);
    }
    final File copiedVelox4jLib = workDir.toPath().resolve(velox4jLibFile.name()).toFile();
    Preconditions.checkState(
        copiedVelox4jLib.isFile(), "Velox4J library not copied to work directory");
    System.load(copiedVelox4jLib.getAbsolutePath());
    System.out.printf("All required libraries were successfully loaded.%n");
  }
}
