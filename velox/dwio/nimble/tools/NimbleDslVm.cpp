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
#include "velox/dwio/nimble/tools/NimbleDslVm.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_map>

namespace facebook::nimble::tools {

std::string toUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

std::string trim(const std::string& s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::vector<std::string> tokenize(const std::string& input) {
  std::vector<std::string> tokens;
  std::istringstream stream{input};
  std::string token;
  while (stream >> token) {
    // Strip trailing semicolons and commas.
    while (!token.empty() && (token.back() == ';' || token.back() == ',')) {
      token.pop_back();
    }
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

Keyword toKeyword(const std::string& token) {
  static const std::unordered_map<std::string, Keyword> kKeywords = {
      {"QUIT", Keyword::Quit},       {"EXIT", Keyword::Exit},
      {"HELP", Keyword::Help},       {"DESCRIBE", Keyword::Describe},
      {"SELECT", Keyword::Select},   {"SHOW", Keyword::Show},
      {"SCHEMA", Keyword::Schema},   {"INFO", Keyword::Info},
      {"STATS", Keyword::Stats},     {"STRIPES", Keyword::Stripes},
      {"STREAMS", Keyword::Streams}, {"ENCODING", Keyword::Encoding},
      {"INDEX", Keyword::Index},     {"HISTOGRAM", Keyword::Histogram},
      {"CONTENT", Keyword::Content}, {"FILE", Keyword::File},
      {"STRIPE", Keyword::Stripe},   {"OPTIONAL", Keyword::Optional},
      {"LIMIT", Keyword::Limit},     {"OFFSET", Keyword::Offset},
      {"FROM", Keyword::From},       {"LAYOUT", Keyword::Layout},
      {"GROUPS", Keyword::Groups},   {"SECTIONS", Keyword::Sections},
      {"TOP", Keyword::Top},         {"METADATA", Keyword::Metadata},
      {"*", Keyword::Star},
  };
  auto it = kKeywords.find(toUpper(token));
  return it != kKeywords.end() ? it->second : Keyword::Unknown;
}

CompileResult compile(const std::vector<std::string>& tokens) {
  CompileResult result;
  if (tokens.empty()) {
    result.error = "Empty input.";
    return result;
  }

  auto command = toKeyword(tokens[0]);

  switch (command) {
    case Keyword::Quit:
    case Keyword::Exit:
      result.program.push_back({OpCode::Quit, NoOperands{}});
      break;

    case Keyword::Help:
      result.program.push_back({OpCode::Help, NoOperands{}});
      break;

    case Keyword::Describe:
      result.program.push_back({OpCode::Describe, NoOperands{}});
      break;

    case Keyword::Select: {
      SelectOperands ops;
      size_t i = 1;

      // Parse column list until we hit a keyword or end.
      while (i < tokens.size()) {
        auto kw = toKeyword(tokens[i]);
        if (kw == Keyword::Limit || kw == Keyword::Offset ||
            kw == Keyword::Stripe || kw == Keyword::From) {
          break;
        }
        if (kw != Keyword::Star) {
          ops.columns.push_back(tokens[i]);
        }
        ++i;
      }

      // Parse optional clauses.
      while (i < tokens.size()) {
        auto kw = toKeyword(tokens[i]);
        if (kw == Keyword::Limit && i + 1 < tokens.size()) {
          ops.limit = std::stoull(tokens[i + 1]);
          i += 2;
        } else if (kw == Keyword::Offset && i + 1 < tokens.size()) {
          ops.offset = std::stoull(tokens[i + 1]);
          i += 2;
        } else if (kw == Keyword::Stripe && i + 1 < tokens.size()) {
          ops.stripeId = std::stoul(tokens[i + 1]);
          i += 2;
        } else if (kw == Keyword::From) {
          // Accept and ignore FROM clause.
          i += 2;
        } else {
          ++i;
        }
      }

      result.program.push_back({OpCode::Select, std::move(ops)});
      break;
    }

    case Keyword::Show: {
      if (tokens.size() < 2) {
        result.error = "SHOW requires a subcommand. Try HELP.";
        break;
      }
      auto sub = toKeyword(tokens[1]);
      switch (sub) {
        case Keyword::Schema:
          result.program.push_back({OpCode::ShowSchema, NoOperands{}});
          break;

        case Keyword::Info:
          result.program.push_back({OpCode::ShowInfo, NoOperands{}});
          break;

        case Keyword::Stats:
          result.program.push_back({OpCode::ShowStats, NoOperands{}});
          break;

        case Keyword::Stripes:
          if (tokens.size() >= 3 && toKeyword(tokens[2]) == Keyword::Metadata) {
            result.program.push_back(
                {OpCode::ShowStripesMetadata, NoOperands{}});
          } else {
            result.program.push_back({OpCode::ShowStripes, NoOperands{}});
          }
          break;

        case Keyword::Streams: {
          ShowStreamsOperands ops;
          if (tokens.size() >= 4 && toKeyword(tokens[2]) == Keyword::Stripe) {
            ops.stripeId = std::stoul(tokens[3]);
          }
          result.program.push_back({OpCode::ShowStreams, std::move(ops)});
          break;
        }

        case Keyword::Encoding: {
          ShowEncodingOperands ops;
          if (tokens.size() >= 4 && toKeyword(tokens[2]) == Keyword::Stripe) {
            ops.stripeId = std::stoul(tokens[3]);
          }
          result.program.push_back({OpCode::ShowEncoding, std::move(ops)});
          break;
        }

        case Keyword::Index:
          result.program.push_back({OpCode::ShowIndex, NoOperands{}});
          break;

        case Keyword::Histogram: {
          ShowHistogramOperands ops;
          size_t i = 2;
          while (i < tokens.size()) {
            auto kw = toKeyword(tokens[i]);
            if (kw == Keyword::Top) {
              ops.topLevel = true;
              ++i;
            } else if (kw == Keyword::Stripe && i + 1 < tokens.size()) {
              ops.stripeId = std::stoul(tokens[i + 1]);
              i += 2;
            } else {
              ++i;
            }
          }
          result.program.push_back({OpCode::ShowHistogram, std::move(ops)});
          break;
        }

        case Keyword::Content: {
          if (tokens.size() < 3) {
            result.error = "SHOW CONTENT requires a stream ID. Try HELP.";
          } else {
            ShowContentOperands ops;
            ops.streamId = std::stoul(tokens[2]);
            if (tokens.size() >= 5 && toKeyword(tokens[3]) == Keyword::Stripe) {
              ops.stripeId = std::stoul(tokens[4]);
            }
            result.program.push_back({OpCode::ShowContent, std::move(ops)});
          }
          break;
        }

        case Keyword::File:
          if (tokens.size() >= 3 && toKeyword(tokens[2]) == Keyword::Layout) {
            result.program.push_back({OpCode::ShowFileLayout, NoOperands{}});
          } else {
            result.error =
                "Unknown SHOW FILE subcommand. Did you mean SHOW FILE LAYOUT?";
          }
          break;

        case Keyword::Stripe:
          if (tokens.size() >= 3 && toKeyword(tokens[2]) == Keyword::Groups) {
            result.program.push_back({OpCode::ShowStripeGroups, NoOperands{}});
          } else {
            result.error =
                "Unknown SHOW STRIPE subcommand. Did you mean SHOW STRIPE GROUPS?";
          }
          break;

        case Keyword::Optional:
          if (tokens.size() >= 3 && toKeyword(tokens[2]) == Keyword::Sections) {
            result.program.push_back(
                {OpCode::ShowOptionalSections, NoOperands{}});
          } else {
            result.error =
                "Unknown SHOW OPTIONAL subcommand. Did you mean SHOW OPTIONAL SECTIONS?";
          }
          break;

        default:
          result.error =
              "Unknown SHOW subcommand: " + tokens[1] + ". Try HELP.";
          break;
      }
      break;
    }

    default:
      result.error = "Unknown command: " + tokens[0] + ". Try HELP.";
      break;
  }

  return result;
}

void printHelp(std::ostream& out, bool color) {
  const char* bold = color ? "\033[1m" : "";
  const char* yellow = color ? "\033[33m" : "";
  const char* cyan = color ? "\033[36m" : "";
  const char* dim = color ? "\033[2m" : "";
  const char* reset = color ? "\033[0m" : "";

  out << bold << "Commands" << reset << dim
      << " (case-insensitive, trailing semicolons optional):" << reset
      << std::endl;
  out << std::endl;

  out << "  " << yellow << "SELECT" << reset
      << " * [LIMIT n] [OFFSET n] [STRIPE s]" << std::endl;
  out << "  " << yellow << "SELECT" << reset
      << " col1, col2 [LIMIT n] [OFFSET n] [STRIPE s]" << std::endl;
  out << "      Read and display row data. Default LIMIT is 20." << std::endl;
  out << dim << "      Examples:" << std::endl;
  out << "        SELECT *" << std::endl;
  out << "        SELECT name, age LIMIT 5" << std::endl;
  out << "        SELECT * LIMIT 10 OFFSET 100" << std::endl;
  out << "        SELECT * LIMIT 50 STRIPE 0" << reset << std::endl;
  out << std::endl;

  out << "  " << yellow << "DESCRIBE" << reset << std::endl;
  out << "      Show top-level column names, Velox types, and Nimble stream"
      << std::endl;
  out << "      offsets in a table." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW SCHEMA" << reset << std::endl;
  out << "      Show the full Nimble schema tree including nested types"
      << std::endl;
  out << "      (arrays, maps, rows, flat maps) with stream offsets."
      << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW INFO" << reset << std::endl;
  out << "      Show file-level metadata: Nimble version, file size, checksum,"
      << std::endl;
  out << "      stripe count, row count, and user-defined metadata."
      << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW STATS" << reset << std::endl;
  out << "      Show per-column statistics: value count, null count, min/max,"
      << std::endl;
  out << "      logical size, and physical size. Requires the file to have"
      << std::endl;
  out << "      been written with vectorized stats enabled." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW STRIPES" << reset << std::endl;
  out << "      Show stripe-level information: stripe ID, byte offset,"
      << std::endl;
  out << "      byte size, and row count for each stripe." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW STREAMS" << reset << " [STRIPE s]"
      << std::endl;
  out << "      Show stream-level information: stream ID, byte offset,"
      << std::endl;
  out << "      byte size, item count, and stream label. Optionally"
      << std::endl;
  out << "      filter to a single stripe." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW ENCODING" << reset << " [STRIPE s]"
      << std::endl;
  out << "      Show per-stream encoding tree: encoding types, data types,"
      << std::endl;
  out << "      and compression used for each stream. Optionally filter"
      << std::endl;
  out << "      to a single stripe." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW INDEX" << reset << std::endl;
  out << "      Show index information: index columns, sort orders,"
      << std::endl;
  out << "      index group metadata, and key stream regions." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW HISTOGRAM" << reset << " [TOP] [STRIPE s]"
      << std::endl;
  out << "      Show encoding type distribution with instance counts and"
      << std::endl;
  out << "      storage percentage. Use TOP to show only top-level encodings."
      << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW CONTENT" << reset << " <stream_id> [STRIPE s]"
      << std::endl;
  out << "      Dump raw decoded stream values for the given stream ID."
      << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW FILE LAYOUT" << reset << std::endl;
  out << "      Show the physical file layout: offsets, sizes, and"
      << std::endl;
  out << "      compression for each section." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW STRIPES METADATA" << reset << std::endl;
  out << "      Show stripes metadata section details: offset, size,"
      << std::endl;
  out << "      and compression type." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW STRIPE GROUPS" << reset << std::endl;
  out << "      Show stripe groups metadata: group ID, offset, size,"
      << std::endl;
  out << "      and compression type." << std::endl;
  out << std::endl;

  out << "  " << yellow << "SHOW OPTIONAL SECTIONS" << reset << std::endl;
  out << "      Show optional sections: name, compression, offset, and size."
      << std::endl;
  out << std::endl;

  out << "  " << cyan << "HELP" << reset << std::endl;
  out << "      Show this help message." << std::endl;
  out << std::endl;

  out << "  " << cyan << "QUIT" << reset << " | " << cyan << "EXIT" << reset
      << " | Ctrl-D" << std::endl;
  out << "      Exit the REPL." << std::endl;
}

bool NimbleDslVm::execute(const std::vector<Instruction>& program) {
  for (const auto& instr : program) {
    switch (instr.opcode) {
      case OpCode::Quit:
        return false;

      case OpCode::Help:
        printHelp(out_, enableColors_);
        break;

      case OpCode::Describe:
        lib_.describe();
        break;

      case OpCode::Select: {
        const auto& ops = std::get<SelectOperands>(instr.operands);
        lib_.select(ops.columns, ops.limit, ops.offset, ops.stripeId);
        break;
      }

      case OpCode::ShowSchema:
        lib_.showSchema();
        break;

      case OpCode::ShowInfo:
        lib_.showInfo();
        break;

      case OpCode::ShowStats:
        lib_.showStats();
        break;

      case OpCode::ShowStripes:
        lib_.showStripes();
        break;

      case OpCode::ShowStreams: {
        const auto& ops = std::get<ShowStreamsOperands>(instr.operands);
        lib_.showStreams(ops.stripeId);
        break;
      }

      case OpCode::ShowEncoding: {
        const auto& ops = std::get<ShowEncodingOperands>(instr.operands);
        lib_.showEncoding(ops.stripeId);
        break;
      }

      case OpCode::ShowIndex:
        lib_.showIndex();
        break;

      case OpCode::ShowHistogram: {
        const auto& ops = std::get<ShowHistogramOperands>(instr.operands);
        lib_.showHistogram(ops.topLevel, ops.stripeId);
        break;
      }

      case OpCode::ShowContent: {
        const auto& ops = std::get<ShowContentOperands>(instr.operands);
        lib_.showContent(ops.streamId, ops.stripeId);
        break;
      }

      case OpCode::ShowFileLayout:
        lib_.showFileLayout();
        break;

      case OpCode::ShowStripesMetadata:
        lib_.showStripesMetadata();
        break;

      case OpCode::ShowStripeGroups:
        lib_.showStripeGroupsMetadata();
        break;

      case OpCode::ShowOptionalSections:
        lib_.showOptionalSections();
        break;
    }
  }
  return true;
}

} // namespace facebook::nimble::tools
