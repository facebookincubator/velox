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

package com.facebook.velox4j.test.dataset.tpch;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.io.FileUtils;

import com.facebook.velox4j.exception.VeloxException;
import com.facebook.velox4j.test.dataset.TestDataFile;
import com.facebook.velox4j.type.RowType;

/**
 * A TPC-H Parquet SF0.1 dataset downloaded from GitHub repo velox4j/tpc-data. Note, this is a
 * temporary solution for generating TPC-H data for testing.
 *
 * <p>TODO: We should implement a local TPC-H data generator in Java instead. Velox4J itself could
 * be used for this workload as well.
 *
 * <p>The utility relies on internet access for downloading the data files. Set the following JVM
 * options if you are behind an HTTP proxy:
 *
 * <p>-Dhttp.proxyHost=your.proxy.host -Dhttp.proxyPort=8080 -Dhttps.proxyHost=your.proxy.host
 * -Dhttps.proxyPort=8080
 *
 * <p>If you want to prepare the data manually before running Velox4J's tests, execute the following
 * commands in advance:
 *
 * <pre>
 * 1. cd /tmp
 * 2. wget https://github.com/velox4j/tpc-data/releases/download/v0.1.0/tpch-parquet-sf0.1.tar.gz
 * 3. tar -xvf tpch-parquet-sf0.1.tar.gz
 * 4. ls -l tpch-parquet-sf0.1 # Check the extracted data.
 * </pre>
 */
class DownloadedTpchDataset implements TpchDataset {
  private static final String DOWNLOAD_LINK =
      "https://github.com/velox4j/tpc-data/releases/download/v0.1.0/tpch-parquet-sf0.1.tar.gz";
  private static final String SHA256 =
      "705f97c32ff4f6e15b8ca84cad58c3900e2047ebb74e103c5b984cee2417aabb";
  private static final String DIRECTORY = "tpch-parquet-sf0.1";
  private static final DownloadedTpchDataset INSTANCE = new DownloadedTpchDataset();

  private final Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"));
  private final Path archivePath = tempDir.resolve("tpch-parquet-sf0.1.tar.gz");
  private final Path extractedDir = tempDir.resolve(DIRECTORY);

  static DownloadedTpchDataset get() {
    return INSTANCE;
  }

  private DownloadedTpchDataset() {
    if (isDownloaded()) {
      System.out.println("TPC-H dataset already exists. Skipping the downloading process...");
      return;
    }
    download();
  }

  private boolean isDownloaded() {
    // Check if the directory exists.
    // Note, we don't check the files so far. If the data is broken, try manually deleting the data
    // and rerun all tests.
    return Files.isDirectory(extractedDir);
  }

  public void download() {
    try {
      System.out.println(
          "Downloading TPC-H Parquet data (this could take a few minutes to finish for the first time)...");
      downloadFile(DOWNLOAD_LINK, archivePath);
      verifySha256(archivePath, SHA256);
      System.out.println("Extracting archive...");
      extractTarGz(archivePath, tempDir);
    } catch (IOException e) {
      try {
        FileUtils.deleteDirectory(extractedDir.toFile());
      } catch (IOException ex) {
        throw new VeloxException("Failed to delete the TPC-H data directory: " + extractedDir, ex);
      }
      throw new VeloxException("Failed to prepare TPC-H dataset: " + e.getMessage(), e);
    }
  }

  private void downloadFile(String url, Path target) throws IOException {
    final URL website = new URL(url);
    try (InputStream in = website.openStream()) {
      Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  private void verifySha256(Path file, String expectedHash) {
    try (InputStream fis = Files.newInputStream(file)) {
      final MessageDigest digest = MessageDigest.getInstance("SHA-256");
      final byte[] buf = new byte[8192];
      int read;
      while ((read = fis.read(buf)) > 0) {
        digest.update(buf, 0, read);
      }
      final StringBuilder sb = new StringBuilder();
      for (byte b : digest.digest()) {
        sb.append(String.format("%02x", b));
      }
      final boolean isMatched = sb.toString().equalsIgnoreCase(expectedHash);
      if (!isMatched) {
        throw new VeloxException("SHA256 checksum does not match");
      }
    } catch (Exception e) {
      throw new VeloxException(e);
    }
  }

  private void extractTarGz(Path archive, Path destDir) throws IOException {
    try (GZIPInputStream gis = new GZIPInputStream(Files.newInputStream(archive));
        TarArchiveInputStream tis = new TarArchiveInputStream(gis)) {
      TarArchiveEntry entry;
      while ((entry = tis.getNextEntry()) != null) {
        final Path outPath = destDir.resolve(entry.getName());
        if (entry.isDirectory()) {
          Files.createDirectories(outPath);
        } else {
          Files.createDirectories(outPath.getParent());
          try (OutputStream out = Files.newOutputStream(outPath)) {
            final byte[] buffer = new byte[8192];
            int len;
            while ((len = tis.read(buffer)) > 0) {
              out.write(buffer, 0, len);
            }
          }
        }
      }
    }
  }

  @Override
  public List<TpchTableName> listFiles() {
    return Arrays.asList(TpchTableName.values());
  }

  @Override
  public TestDataFile get(TpchTableName tpchTableName) {
    return new TestDataFile() {
      @Override
      public RowType schema() {
        return tpchTableName.schema();
      }

      @Override
      public File file() {
        return extractedDir.resolve(tpchTableName.relativePath()).toFile();
      }
    };
  }
}
