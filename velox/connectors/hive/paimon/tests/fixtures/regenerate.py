#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate offline fixtures using locally built, fixed-version Paimon jars."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paimon", type=Path, required=True)
    parser.add_argument("--maven-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paimon = args.paimon.resolve()
    repo = args.maven_repo.resolve()
    work = args.work_dir.resolve()
    output = args.output.resolve()
    if work.exists() or output.exists():
        parser.error("Use fresh work and output directories")

    modules = ["api", "common", "core", "format", "codegen-loader"]
    jars = [
        paimon / f"paimon-{name}/target/paimon-{name}-2.2-SNAPSHOT.jar"
        for name in modules
    ]
    # Runtime dependencies of the fixed Paimon profile. Paimon jars themselves
    # always come from the supplied source build, never from a snapshot repo.
    dependencies = [
        "org.slf4j:slf4j-api:1.7.32",
        "org.xerial.snappy:snappy-java:1.1.10.8",
        "at.yawk.lz4:lz4-java:1.10.4",
        "org.apache.hadoop:hadoop-common:2.8.5",
        "org.apache.hadoop:hadoop-auth:2.8.5",
        "org.apache.hadoop:hadoop-mapreduce-client-core:2.8.5",
        "com.google.guava:guava:11.0.2",
        "commons-logging:commons-logging:1.1.3",
        "commons-collections:commons-collections:3.2.2",
        "commons-lang:commons-lang:2.6",
        "commons-configuration:commons-configuration:1.6",
        "commons-io:commons-io:2.4",
    ]
    dependencies += [
        f"org.apache.logging.log4j:{name}:2.25.5"
        for name in ["log4j-api", "log4j-core", "log4j-1.2-api", "log4j-slf4j-impl"]
    ]
    for dependency in dependencies:
        group, name, version = dependency.split(":")
        jars.append(
            repo / group.replace(".", "/") / name / version / f"{name}-{version}.jar"
        )
    for jar in jars:
        if not jar.is_file():
            parser.error(f"Missing dependency; build Paimon first: {jar}")

    work.mkdir(parents=True)
    classes = work / "classes"
    classes.mkdir()
    temp = work / "tmp"
    temp.mkdir()
    classpath = os.pathsep.join(map(str, jars))
    subprocess.run(
        [
            "javac",
            "--release",
            "8",
            "-cp",
            classpath,
            "-d",
            str(classes),
            str(Path(__file__).with_name("GenerateAppend.java")),
        ],
        check=True,
    )
    subprocess.run(
        [
            "java",
            f"-Djava.io.tmpdir={temp}",
            "-cp",
            str(classes) + os.pathsep + classpath,
            "GenerateAppend",
            str(work / "warehouse"),
            str(output),
        ],
        check=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    files = [manifest["emptyFile"]]
    for snapshot in ["append", "cow"]:
        for split in manifest[snapshot]["splits"]:
            files.extend(split["files"])
    for file in files:
        contents = (output / file["filePath"]).read_bytes()
        assert len(contents) == file["fileSize"]
        assert hashlib.sha256(contents).hexdigest() == file["sha256"]
    schema = json.loads((output / "schema.json").read_text())
    schema["options"].pop("path", None)
    for name, value in [("schema", schema), ("manifest", manifest)]:
        (output / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )
    print(f"Generated and checked fixtures: {output}")


if __name__ == "__main__":
    main()
