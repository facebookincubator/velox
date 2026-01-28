# Workflow summary and overview

## Summary

This file describes the workflow for the project. It is a living document that will be updated as the project progresses. It is intended to be a high-level overview of the workflow and not a detailed description of each step.

Specifically, it covers the CI compiler and feature coverage. There are limited resources available and an attempt is made to cover as many combinations in a single workflow as possible.

## Overview

### Linux

Files: `.github/workflows/linux-build.yml` and `.github/workflows/linux-build-base.yml`

The main workflow jobs are implemented in the `linux-build-base.yml` file. This file is included by the `linux-build.yml` file. The `linux-build.yml` file is the main workflow file that is triggered by a push to the main branch or a pull request. It is also triggered by a schedule to run the workflow on a regular basis. The `linux-build-base.yml` file is included by the `linux-build.yml` file to avoid code duplication.

#### PR and merge to main trigger coverage

| Name | Build type | Adapters | GPU | Compiler | Image | Size |
| ---- | ---------- | -------- | --- | -------- | ----- | ---- |
| adapters | release | Y | Y | GCC14 | Centos9 | 32 GB / 16 CPU |
| ubuntu-debug | debug | N | GCC11 | None | 16GB / 8 CPU |
| fedora-debug | debug | N | GCC14(++)/15 | Fedora | 32 GB / 16 CPU |
| asan-ubsan-adapters | debug with -O1 | X (-FAISS) | N | Clang 21 | Centos9 | 32 GB / 16 CPU |

#### Schedule coverage

| Name | Build type | Adapters | GPU | Compiler | Image | Size |
| ---- | ---------- | -------- | --- | -------- | ----- | ---- |
| adapters | release | Y | Y | Clang 15 | Centos9 | 32 GB / 16 CPU |
| ubuntu-debug | debug | N | Clang 15 | None | 16GB / 8 CPU |
| fedora-debug | debug | N | GCC14(++)/15 | Fedora | 32 GB / 16 CPU |
| asan-ubsan-adapters | debug with -O1 | X (-FAISS) | N | Clang 21 | Centos9 | 32 GB / 16 CPU |

### macOS

File: `.github/workflows/mac-build.yml`

#### PR and merge to main trigger coverage

| Name | Build type | Adapters | GPU | Compiler | Image |
| ---- | ---------- | -------- | --- | -------- | ----- |
| macos-build | release, debug | N | N | Apple Clang (arm) | github macos-15 |
