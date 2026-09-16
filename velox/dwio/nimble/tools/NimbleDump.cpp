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
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>

#include "common/base/BuildInfo.h"
#include "common/config/Flags.h"
#include "common/init/light.h"
#include "folly/Singleton.h"
#include "folly/cli/NestedCommandLineApp.h"
#include "folly/logging/Init.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/nimble/tools/NimbleDumpLib.h"

using namespace facebook;
namespace po = ::boost::program_options;

namespace {
template <typename T>
std::optional<T> getOptional(const po::variable_value& val) {
  return val.empty() ? std::nullopt : std::optional<T>(val.as<T>());
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

  // Enable colored output if we are running in a terminal
  bool enableColors = isColorfulTty();

  std::string version{BuildInfo::getBuildTimeISO8601()};
  if (!version.empty()) {
    auto buildRevision = BuildInfo::getBuildRevision();
    if (buildRevision && buildRevision[0] != '\0') {
      version += folly::to<std::string>(" [", buildRevision, "]");
    }
  }

  folly::NestedCommandLineApp app{"", std::move(version)};
  int style = po::command_line_style::default_style;
  style &= ~po::command_line_style::allow_guessing;
  app.setOptionStyle(static_cast<po::command_line_style::style_t>(style));

  auto makePositionalArgs = []() {
    po::positional_options_description positionalArgs;
    positionalArgs.add("file", /*max_count*/ 1);
    return positionalArgs;
  };

  app.addCommand(
         "info",
         "<file>",
         "Print file information",
         "Prints file information from the file footer.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitInfo();
         },
         makePositionalArgs())
      .add_options()(
          "file",
          po::value<std::string>()->required(),
          "Nimble file path. Can be a local path or a Warm Storage path.");

  app.addCommand(
         "file_layout",
         "<file>",
         "Print overall layout of the file",
         "Print overall layout of the file",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitFileLayout(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on

  app.addCommand(
         "schema",
         "<file>",
         "Print file schema",
         "Prints the file schema tree.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitSchema(!options["full"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
            (
                "file",
                po::value<std::string>()->required(),
                "Nimble file path. Can be a local path or a Warm Storage path."
            )(
                "full,f",
                po::bool_switch()->default_value(false),
                "Emit full flat map schemas. Default is to collapse flat map schemas."
            );
  // clang-format on

  app.addCommand(
         "stripes",
         "<file>",
         "Print stripe information",
         "Prints detailed stripe information.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitStripes(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on

  app.addCommand(
         "streams",
         "<file>",
         "Print stream information",
         "Prints detailed stream information.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitStreams(
                   options["no_header"].as<bool>(),
                   options["labels"].as<bool>(),
                   options["raw_size"].as<bool>(),
                   options["inmap_stream"].as<bool>(),
                   getOptional<uint32_t>(options["stripe"]));
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
            (
                "file",
                po::value<std::string>()->required(),
                "Nimble file path. Can be a local path or a Warm Storage path."
            )(
                "stripe,s",
                po::value<uint32_t>(),
                "Limit output to a single stripe with the provided stripe id. "
                "Default is to print streams for all stripes."
            )(
                "raw_size,r",
                po::bool_switch()->default_value(false),
                "Print raw stream sizes. Default is to omit raw stream sizes."
            )(
                "no_header,n",
                po::bool_switch()->default_value(false),
                "Don't print column names. Default is to include column names."
            )(
                "labels,l",
                po::bool_switch()->default_value(false),
                "Include stream labels. Lables provide a readable path from the "
                "root node to the stream, as they appear in the schema tree."
            )(
                "inmap_stream,i",
                po::bool_switch()->default_value(false),
                "Include InMap stream indicator."
            );
  // clang-format on

  app.addCommand(
         "histogram",
         "<file>",
         "Print encoding histogram",
         "Prints encoding histogram, counting how many times each encoding "
         "appears in the file.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitHistogram(
                   options["root_only"].as<bool>(),
                   options["no_header"].as<bool>(),
                   getOptional<uint32_t>(options["stripe"]));
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
            (
                "file",
                po::value<std::string>()->required(),
                "Nimble file path. Can be a local path or a Warm Storage path."
            )(
                "stripe,s",
                po::value<uint32_t>(),
                "Limit analysis to a single stripe with the provided stripe id. "
                "Default is to analyze encodings in all stripes."
            )(
                "root_only,r",
                po::bool_switch()->default_value(false),
                "Include only root (top level) encodings in histogram. "
                "Default is to analyze full encoding trees."
            )(
                "no_header,n",
                po::bool_switch()->default_value(false),
                "Don't print column names. Default is to include column names."
            );
  // clang-format on

  app.addCommand(
         "content",
         "<file>",
         "Print the content of a stream",
         "Prints the materialized content (actual values) of a stream.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitContent(
                   options["stream"].as<uint32_t>(),
                   getOptional<uint32_t>(options["stripe"]),
                   options["separator"].as<std::string>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
            (
                "file",
                po::value<std::string>()->required(),
                "Nimble file path. Can be a local path or a Warm Storage path."
            )(
                "stream",
                po::value<uint32_t>()->required(),
                "The content of this stream id will be emitted."
            )(
                "stripe",
                po::value<uint32_t>(),
                "Limit output to a single stripe with the provided stripe id. "
                "Default is to output stream content across in all stripes."
            )(
                "separator",
                po::value<std::string>()->default_value("\n"),
                "Record separator. Default is \\n."
            );
  // clang-format on

  app.addCommand(
         "binary",
         "<file>",
         "Dumps stream binary content",
         "Dumps stream binary content to a file.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitBinary(
                   [path = options["output"].as<std::string>()]() {
                     return std::make_unique<std::ofstream>(
                         path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
                   },
                   options["stream"].as<uint32_t>(),
                   options["stripe"].as<uint32_t>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
            (
                "file",
                po::value<std::string>()->required(),
                "Nimble file path. Can be a local path or a Warm Storage path."
            )(
                "output,o",
                po::value<std::string>()->required(),
                "Output file path."
            )(
                "stream",
                po::value<uint32_t>()->required(),
                "The content of this stream id will be dumped to the output file."
            )(
                "stripe",
                po::value<uint32_t>()->required(),
                "Dumps the stream from this stripe id."
            );
  // clang-format on

  app.addCommand(
         "layout",
         "<file>",
         "Dumps layout file",
         "Dumps captured layout file content.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitLayout(
                   options["no_header"].as<bool>(),
                   !options["uncompressed"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
          .add_options()
              (
                  "file",
                  po::value<std::string>()->required(),
                  "Encoding layout file path."
              )(
                "no_header,n",
                po::bool_switch()->default_value(false),
                "Don't print column names. Default is to include column names."
              )(
                "uncompressed,u",
                po::bool_switch()->default_value(false),
                "Is the layout file uncompressed. Default is false, which means "
                "the layout file is compressed."
              );
  // clang-format on

  app.addCommand(
         "stripes_metadata",
         "<file>",
         "Print stripes metadata information",
         "Prints stripes metadata information as referenced by the footer.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitStripesMetadata(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on

  app.addCommand(
         "stripe_groups_metadata",
         "<file>",
         "Print stripe groups metadata information",
         "Prints stripe groups information as referenced by the footer.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitStripeGroupsMetadata(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on
  app.addCommand(
         "optional_sections_metadata",
         "<file>",
         "Print optional sections information",
         "Prints optional sections information as referenced by the footer.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitOptionalSectionsMetadata(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on

  app.addCommand(
         "stats",
         "<file>",
         "Print file stats",
         "Prints statistics information from the file. "
         "Automatically detects and prints vectorized column stats if available, "
         "otherwise falls back to legacy stats.",
         [enableColors](
             const po::variables_map& options,
             const std::vector<std::string>& /*args*/) {
           nimble::tools::NimbleDumpLib{
               options["file"].as<std::string>(), enableColors, std::cout}
               .emitStats(options["no_header"].as<bool>());
         },
         makePositionalArgs())
      // clang-format off
        .add_options()
        (
            "file",
            po::value<std::string>()->required(),
            "Nimble file path. Can be a local path or a Warm Storage path."
        )(
            "no_header,n",
            po::bool_switch()->default_value(false),
            "Don't print column names. Default is to include column names."
        );
  // clang-format on

  app.addAlias("i", "info");
  app.addAlias("b", "binary");
  app.addAlias("c", "content");

  try {
    return app.run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}

// Initialize dummy Velox stats reporter
folly::Singleton<facebook::velox::BaseStatsReporter> reporter([]() {
  return new facebook::velox::DummyStatsReporter();
});
