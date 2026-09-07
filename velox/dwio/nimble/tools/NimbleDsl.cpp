/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <cstring>
#include <iostream>
#include <string>

#include "common/config/Flags.h"
#include "common/init/light.h"
#include "folly/Singleton.h"
#include "folly/logging/Init.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/nimble/tools/NimbleDslVm.h"

using namespace facebook;

namespace {

void printUsage(std::ostream& out) {
  out << "Usage: nimble_dsl <file>" << std::endl;
  out << std::endl;
  out << "Interactive SQL-like REPL for inspecting Nimble files." << std::endl;
  out << "Opens the given Nimble file and presents a prompt where you can"
      << std::endl;
  out << "run commands to query data, view schema, and inspect metadata."
      << std::endl;
  out << std::endl;
  out << "Type HELP at the prompt for a list of available commands."
      << std::endl;
}

bool isColorfulTty() {
  auto isTty = isatty(fileno(stdout));
  if (!isTty) {
    return false;
  }
  auto term = std::getenv("TERM");
  return !(term == nullptr || term[0] == '\0' || strcmp(term, "dumb") == 0);
}

} // namespace

FOLLY_INIT_LOGGING_CONFIG("CRITICAL");
int main(int argc, char* argv[]) {
  facebook::config::Flags::overrideDefault("minloglevel", "5");

  auto init = init::InitFacebookLight{
      &argc, &argv, folly::InitOptions().useGFlags(false)};

  velox::filesystems::registerLocalFileSystem();

  if (argc < 2) {
    printUsage(std::cerr);
    return EXIT_FAILURE;
  }

  std::string filePath = argv[1];

  if (filePath == "--help" || filePath == "-h") {
    printUsage(std::cout);
    return EXIT_SUCCESS;
  }

  bool enableColors = isColorfulTty();

  if (enableColors) {
    // clang-format off
    std::cout << "\033[1m\033[36m"
      "  _   _ _           _     _      \n"
      " | \\ | (_)_ __ ___ | |__ | | ___ \n"
      " |  \\| | | '_ ` _ \\| '_ \\| |/ _ \\\n"
      " | |\\  | | | | | | | |_) | |  __/\n"
      " |_| \\_|_|_| |_| |_|_.__/|_|\\___|\n"
      "\033[0m" << std::endl;
    // clang-format on
    std::cout << " Interactive SQL-like shell for Nimble files." << std::endl;
    std::cout << " Type \033[33mHELP\033[0m for commands, \033[33mQUIT\033[0m"
                 " to exit."
              << std::endl;
    std::cout << std::endl;
  } else {
    // clang-format off
    std::cout <<
      "  _   _ _           _     _      \n"
      " | \\ | (_)_ __ ___ | |__ | | ___ \n"
      " |  \\| | | '_ ` _ \\| '_ \\| |/ _ \\\n"
      " | |\\  | | | | | | | |_) | |  __/\n"
      " |_| \\_|_|_| |_| |_|_.__/|_|\\___|\n"
      << std::endl;
    // clang-format on
    std::cout << " Interactive SQL-like shell for Nimble files." << std::endl;
    std::cout << " Type HELP for commands, QUIT to exit." << std::endl;
    std::cout << std::endl;
  }

  nimble::tools::NimbleDslLib lib{std::cout, enableColors, filePath};
  nimble::tools::NimbleDslVm vm{lib, std::cout, enableColors};

  std::string line;
  while (true) {
    if (enableColors) {
      std::cout << "\033[1m\033[32mnimble>\033[0m " << std::flush;
    } else {
      std::cout << "nimble> " << std::flush;
    }
    if (!std::getline(std::cin, line)) {
      // EOF (Ctrl-D).
      std::cout << std::endl;
      break;
    }

    auto trimmed = nimble::tools::trim(line);
    if (trimmed.empty()) {
      continue;
    }

    auto tokens = nimble::tools::tokenize(trimmed);
    if (tokens.empty()) {
      continue;
    }

    try {
      auto compiled = nimble::tools::compile(tokens);
      if (!compiled.error.empty()) {
        std::cerr << "\033[31m" << compiled.error << "\033[0m" << std::endl;
        continue;
      }
      if (!vm.execute(compiled.program)) {
        break; // Quit instruction.
      }
    } catch (const std::exception& e) {
      std::cerr << (enableColors ? "\033[31m" : "") << "Error: " << e.what()
                << (enableColors ? "\033[0m" : "") << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::Singleton<facebook::velox::BaseStatsReporter> reporter([]() {
  return new facebook::velox::DummyStatsReporter();
});
