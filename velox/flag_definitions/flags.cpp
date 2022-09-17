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

#include "velox/flag_definitions/flags.h"

namespace facebook::velox::flags {
namespace po = boost::program_options;

VeloxFlags::VeloxFlags() {
  po::options_description options = addOptions();
  po::basic_parsed_options<char> parsedOptions(&options);
  po::store(parsedOptions, vm_);
  po::notify(vm_);
}

void VeloxFlags::init(int argc, const char* const argv[]) {
  po::store(po::parse_command_line(argc, argv, addOptions()), vm_);
  po::notify(vm_);
}

void VeloxFlags::init(std::map<std::string, std::string> options) {
  std::stringstream sout;

  for (auto kv = options.begin(); kv != options.end(); kv++) {
    sout << kv->first << "=" << kv->second << std::endl;
  }

  vm_.clear();
  po::store(po::parse_config_file(sout, addOptions()), vm_);
  po::notify(vm_);
}

po::options_description VeloxFlags::addOptions() {
  po::options_description options("Velox flags.");
  options.add_options()(
      "avx2",
      po::value<bool>()->default_value(true),
      "Enables use of AVX2 when available")(
      "bmi2",
      po::value<bool>()->default_value(true),
      "Enables use of BMI2 when available")(
      "max_block_value_set_length",
      po::value<int32_t>()->default_value(5),
      "Max entries per column that the block meta-record stores for pre-flight "
      "filtering optimization")(
      "memory_usage_aggregation_interval_millis",
      po::value<int32_t>()->default_value(2),
      "Interval to compute aggregate memory usage for all nodes")(
      "velox_memory_pool_mb",
      po::value<int32_t>()->default_value(4 * 1024),
      "Size of file cache/operator working memory in MB")(
      "velox_use_malloc",
      po::value<bool>()->default_value(true),
      "Use malloc for file cache and large operator allocations")(
      "velox_time_allocations",
      po::value<bool>()->default_value(true),
      "Record time and volume for large allocation/free")(
      "velox_exception_system_stacktrace_enabled",
      po::value<bool>()->default_value(true),
      "Enable the stacktrace for system type of VeloxException")(
      "velox_exception_user_stacktrace_enabled",
      po::value<bool>()->default_value(false),
      "Enable the stacktrace for user type of VeloxException")(
      "velox_exception_user_stacktrace_rate_limit_ms",
      po::value<int32_t>()->default_value(0),
      "Min time interval in milliseconds between stack traces captured in"
      " user type of VeloxException; off when set to 0 (the default)")(
      "velox_exception_system_stacktrace_rate_limit_ms",
      po::value<int32_t>()->default_value(0),
      "Min time interval in milliseconds between stack traces captured in"
      " system type of VeloxException; off when set to 0 (the default)");

  return options;
}

VeloxFlags& getInstance() {
  static VeloxFlags instance;
  return instance;
}

} // namespace facebook::velox::flags
