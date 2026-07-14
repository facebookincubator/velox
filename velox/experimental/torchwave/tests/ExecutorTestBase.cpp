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

#include "velox/experimental/torchwave/tests/ExecutorTestBase.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h> // @manual
#include <folly/compression/Compression.h>
#include <glog/logging.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/SerialGraphExecutor.h>

#include <caffe2/caffe2/serialize/file_adapter.h>

#include <torch/nativert/graph/GraphPasses.h>
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

// debug_single_ops is now WaveConfig::debugSingleOps

DEFINE_bool(print_timing, false, "Print timing for wave graph execution");
DEFINE_int32(num_repeats, 1, "Number of timed repetitions for each run type");
DEFINE_bool(
    standalone_kernels,
    false,
    "Treat all operations as standalone kernels");
DEFINE_int32(block_dim, 256, "CUDA thread block size");
DEFINE_int32(
    single_block,
    -1,
    "Force single block grid: -1=auto, 0=multi, 1=single");
DEFINE_int32(cg, -1, "Force cooperative grid: -1=auto, 0=off, 1=on");
DEFINE_bool(wave_only, false, "Skip serial CPU and serial GPU execution");
DEFINE_int32(
    single_ops,
    -1,
    "Debug single ops mode: -1=run normal and single-ops, 0=normal only, 1=single-ops only");
DEFINE_bool(
    standalone_timing,
    false,
    "Print standalone execution times after wave run");
DEFINE_bool(
    print_full_mismatch,
    false,
    "Print full tensor contents for mismatched outputs");
DEFINE_int32(
    list,
    0,
    "List WaveGraph before execution: 0=off, 1=kExprs, 2=kGrids");
DEFINE_int32(trace, 0, "Trace bit mask: 1=nodes, 2=launches");
DEFINE_string(
    print_options,
    "",
    "Comma-separated NodePrinter options: D<n>=maxDepth, L<n>=maxLength, "
    "S=shortNames, V=per-line values, NA=no attributes, VN=value names");
DEFINE_string(
    save_reference_frame,
    "",
    "Path to save execution frame after nativert GPU run");
DEFINE_string(
    reference_frame,
    "",
    "Path to load reference frame for verifying wave intermediates");
DEFINE_bool(
    reference_from_gpu,
    false,
    "When saving a reference frame, capture it from the nativert GPU run "
    "(cpuOnly args moved to CPU via inserted _to_copy) instead of the CPU run");
DEFINE_bool(
    reverify,
    false,
    "Re-verify all previously passed reference values each step to detect corruption");
DEFINE_string(
    kernel_cache_dir,
    "",
    "If non-empty, cache compiled CUDA kernels (cubin) in this directory");
DEFINE_int32(
    max_elementwise_vars,
    7,
    "Max pointer variables in elementwise codegen; beyond this, storage expressions are inlined");
DEFINE_int32(
    out_of_line_expr_size,
    10'000,
    "Character threshold for extracting elementwise subtrees into noinline helpers");
DEFINE_string(
    trace_values,
    "",
    "Comma-separated list of value ids to trace during wave execution");
DEFINE_int32(
    tensor_print_limit,
    100,
    "Max elements printed per tensor when tracing values; 0 for no limit");
DEFINE_bool(
    throw_on_error,
    true,
    "Throw on kernel error; if false, print error after verify instead");
DEFINE_bool(
    no_elementwise_fast_path,
    false,
    "Skip elementwise fast path; always use slow path with complexIdx");
DEFINE_bool(
    continue_after_mismatch,
    false,
    "Log reference mismatches but continue instead of throwing");
DEFINE_bool(
    kernel_debug_output,
    false,
    "Enable device-side debug printfs. Emergency use only");
DEFINE_bool(
    debug_single_ops,
    false,
    "Launch kernel once per block for debugging, waiting after each launch");
DEFINE_bool(
    auto_adjust_cost,
    false,
    "Adjust per-op cost multipliers after each execution based on actual thread block clocks");

DEFINE_bool(
    enable_reuse,
    false,
    "Reuse a value's buffer in place when an op is its unique last use (turn copying ops into in-place ops)");
DEFINE_bool(
    free_intermediates,
    false,
    "Release each ProjectNode's last-use value tensors right after its composite invocation executes, instead of at end-of-graph");

namespace torch::wave {

using Clock = std::chrono::high_resolution_clock;

std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath) {
  auto cwd = std::filesystem::current_path().string();
  if (cwd.size() >= 6 && cwd.compare(cwd.size() - 6, 6, "fbcode") == 0) {
    return cwd + "/" + baseDir + "/" + filePath;
  }
  return cwd + "/" + filePath;
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    throw std::runtime_error("Cannot open " + path);
  }
  auto size = file.tellg();
  file.seekg(0);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

namespace {

void flattenIValue(const c10::IValue& value, std::vector<c10::IValue>& out) {
  if (value.isObject()) {
    auto obj = value.toObject();
    for (size_t i = 0; i < obj->slots().size(); ++i) {
      out.push_back(obj->getSlot(i));
    }
  } else if (value.isTuple()) {
    for (const auto& elem : value.toTupleRef().elements()) {
      out.push_back(elem);
    }
  } else {
    out.push_back(value);
  }
}

void deepFlattenIValue(
    const c10::IValue& value,
    std::vector<c10::IValue>& out) {
  if (value.isNone()) {
    return;
  }
  if (value.isObject()) {
    auto obj = value.toObject();
    for (size_t i = 0; i < obj->slots().size(); ++i) {
      deepFlattenIValue(obj->getSlot(i), out);
    }
  } else if (value.isTuple()) {
    for (const auto& elem : value.toTupleRef().elements()) {
      deepFlattenIValue(elem, out);
    }
  } else {
    out.push_back(value);
  }
}

void fillFrameInputs(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    std::vector<c10::IValue> inputs) {
  const auto& userInputNames = graph.signature().userInputs();
  // Flatten one level to handle Objects/Tuples wrapping the actual inputs.
  std::vector<c10::IValue> flat;
  for (auto& v : inputs) {
    flattenIValue(v, flat);
  }
  // If still too many, flatten one more level.
  while (flat.size() > userInputNames.size()) {
    std::vector<c10::IValue> next;
    bool changed = false;
    for (auto& v : flat) {
      auto before = next.size();
      flattenIValue(v, next);
      if (next.size() - before != 1) {
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    flat = std::move(next);
  }
  size_t flatIdx = 0;
  for (const auto& name : userInputNames) {
    auto* value = graph.tryGetValue(name);
    if (value && flatIdx < flat.size()) {
      frame.setIValue(value->id(), std::move(flat[flatIdx++]));
    }
  }
}

// Fill a frame from inputs already aligned 1:1 with graph.userInputs(), e.g.
// synthetic data generated per user input. Unlike fillFrameInputs this does not
// re-flatten and keys off graph.userInputs() rather than the signature's
// userInputs: for models like the ROO preproc the two lists differ in length
// (graph.userInputs() includes the optional/None leaves the signature omits),
// and only graph.userInputs() aligns positionally with the generated data.
void fillFrameFromUserInputs(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    const std::vector<c10::IValue>& inputs) {
  const auto& userInputs = graph.userInputs();
  for (size_t i = 0; i < userInputs.size() && i < inputs.size(); ++i) {
    if (userInputs[i] && !inputs[i].isNone()) {
      if (inputs[i].isTensor()) {
        frame.setIValue(
            userInputs[i]->id(), c10::IValue(inputs[i].toTensor().clone()));
      } else {
        frame.setIValue(userInputs[i]->id(), inputs[i]);
      }
    }
  }
}

// Removes aten._assert_async runtime data-validation nodes from the graph.
// These guard real-data invariants (e.g. "num_candidates must be all True",
// "labels must equal label presences") that random synthetic data does not
// satisfy, so they fire spurious device-side asserts on a synthetic run. Their
// outputs are unused, so removal is safe; mirrors nativert's RemoveDetach pass.
void stripDataAsserts(nativert::Graph& graph) {
  std::vector<nativert::Node*> toDrop;
  for (auto& node : graph.nodes()) {
    if (node.target() == "torch.ops.aten._assert_async.msg") {
      toDrop.push_back(&node);
    }
  }
  if (toDrop.empty()) {
    return;
  }
  for (auto* node : toDrop) {
    node->destroy();
  }
  graph.renumberValues();
  graph.finalize();
  graph.lint();
  LOG(INFO) << "stripDataAsserts: removed " << toDrop.size()
            << " _assert_async node(s)";
}

// Decompresses a gzipped file to a fresh temp file and returns its path. The
// committed synthetic-graph archives are stored as <name>.pt2.gz because the
// raw graph JSON is far over the repo's per-file size limit; this restores a
// plain .pt2 the PyTorchStreamReader can open. Caller removes the temp file.
std::string decompressGzToTemp(const std::string& gzPath) {
  auto data = readFile(gzPath);
  auto codec =
      folly::compression::getCodec(folly::compression::CodecType::GZIP);
  std::string out =
      codec->uncompress(folly::StringPiece(data.data(), data.size()));
  std::string tmp = "/tmp/torchwave_synth_" + std::to_string(getpid()) + ".pt2";
  std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
  TORCH_CHECK(o.is_open(), "Cannot write temp pt2: ", tmp);
  o.write(out.data(), static_cast<std::streamsize>(out.size()));
  return tmp;
}

} // namespace

std::vector<c10::IValue> loadReferenceValues(const std::string& path) {
  auto data = readFile(path);
  auto ivalue = pickleLoadWithTypes(data);
  std::vector<c10::IValue> values;
  if (ivalue.isList()) {
    for (const auto& item : ivalue.toListRef()) {
      flattenIValue(item, values);
    }
  } else {
    flattenIValue(ivalue, values);
  }
  return values;
}

std::pair<std::vector<c10::IValue>, int64_t> inputsToDevice(
    std::vector<c10::IValue>& inputs) {
  std::vector<at::Tensor> hostTensors;
  std::vector<size_t> tensorIndices;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].isTensor()) {
      hostTensors.push_back(inputs[i].toTensor());
      tensorIndices.push_back(i);
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> deviceTensors;

  auto start = Clock::now();
  tensorsToDevice(hostTensors, deviceTensors, stream);
  stream.wait();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start)
                .count();

  std::vector<c10::IValue> result(inputs.size());
  for (size_t i = 0, ti = 0; i < inputs.size(); ++i) {
    if (ti < tensorIndices.size() && tensorIndices[ti] == i) {
      result[i] = c10::IValue(deviceTensors[ti]);
      ++ti;
    } else {
      result[i] = inputs[i];
    }
  }

  return {std::move(result), us};
}

int32_t insertCpuOnlyCopies(nativert::Graph& graph) {
  // Collect insertion sites first; don't mutate the node list while iterating.
  struct Site {
    nativert::Node* consumer;
    size_t inputIdx;
    nativert::Value* deviceValue;
  };
  std::vector<Site> sites;
  for (auto& node : graph.nodes()) {
    const auto* meta = Registry::metadata(node.target());
    if (!meta) {
      continue;
    }
    auto& inputs = node.inputs();
    for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size();
         ++i) {
      // cpuOnly is only set on tensor args (e.g. tensor_split indices), so a
      // non-null value here is the tensor we must keep on host.
      if (meta->argumentMeta[i].cpuOnly && inputs[i].value) {
        sites.push_back({&node, i, inputs[i].value});
      }
    }
  }

  // For each cpuOnly tensor arg, insert aten._to_copy(self, device=cpu) right
  // before the consumer and repoint only that edge. tensor_split (the only
  // cpuOnly op) reads its indices on the host and returns views of self, so
  // self and the outputs stay on GPU -- no move-back is needed. Lets the
  // generic nativert executor run the graph on GPU (wave handles cpuOnly args
  // itself at runtime; see Launch in CompiledOp.cpp).
  for (const auto& site : sites) {
    auto* copyNode = graph.createNode(
        "torch.ops.aten._to_copy.default", {{"self", site.deviceValue}});
    copyNode->addAttribute({"dtype", torch::nativert::None{}});
    copyNode->addAttribute({"layout", torch::nativert::None{}});
    copyNode->addAttribute({"device", c10::Device(c10::kCPU)});
    copyNode->addAttribute({"pin_memory", torch::nativert::None{}});
    copyNode->addAttribute({"non_blocking", false});
    copyNode->addAttribute({"memory_format", torch::nativert::None{}});
    auto* cpuValue = copyNode->addOutput(
        graph.getUniqueValueName(), site.deviceValue->type());
    graph.insertBefore(copyNode, site.consumer);
    site.consumer->inputs()[site.inputIdx].value = cpuValue;
    site.deviceValue->eraseUser(site.consumer);
    cpuValue->addUser(site.consumer);
  }
  LOG(INFO) << "insertCpuOnlyCopies: inserted " << sites.size()
            << " _to_copy(device=cpu) node(s)";
  return static_cast<int32_t>(sites.size());
}

int32_t rewriteGpuIncompatibleOps(nativert::Graph& graph) {
  // fb::simple_1d_concat has no CUDA implementation -- its CUDA registration is
  // a throwing Dummy
  // (caffe2/torch/fb/sparsenn/cpu_operators/simple_concat.cpp). It is a plain
  // 1-D concat, identical to aten.cat(tensors, dim=0), which does have a CUDA
  // kernel. Mirror wave's rewrite (MoreBuiltins.cpp) so the generic nativert
  // executor can run it on GPU. (The other fb ops in this model -- sigrid_hash,
  // grouped_masked_select_jagged_1d, batch_flip_and_truncate_sparse,
  // group_length_guard_sparse, fused_datafm_merge_and_dedup_by_reference --
  // have real CUDA kernels in sparsenn_operators_gpu and self-register.)
  int32_t rewritten = 0;
  for (auto& node : graph.nodes()) {
    if (node.target() == "torch.ops.fb.simple_1d_concat.default") {
      node.setTarget("torch.ops.aten.cat.default");
      // nativert matches node inputs to schema args by name; rename the
      // TensorList input "inputs" -> "tensors" to match aten::cat's schema.
      if (!node.inputs().empty()) {
        node.inputs()[0].name = "tensors";
      }
      node.addAttribute({"dim", static_cast<int64_t>(0)});
      ++rewritten;
    }
  }
  LOG(INFO) << "rewriteGpuIncompatibleOps: rewrote " << rewritten
            << " fb.simple_1d_concat -> aten.cat.default";
  return rewritten;
}

std::vector<c10::IValue> outputsToHost(
    const std::vector<c10::IValue>& outputs,
    const std::string& /*label*/) {
  std::vector<at::Tensor> deviceTensors;
  std::vector<size_t> tensorIndices;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      deviceTensors.push_back(outputs[i].toTensor());
      tensorIndices.push_back(i);
    }
  }

  facebook::velox::wave::Stream stream;
  std::vector<at::Tensor> hostTensors;
  tensorsToHost(deviceTensors, hostTensors, stream);
  stream.wait();

  std::vector<c10::IValue> result(outputs.size());
  for (size_t i = 0, ti = 0; i < outputs.size(); ++i) {
    if (ti < tensorIndices.size() && tensorIndices[ti] == i) {
      result[i] = hostTensors[ti];
      ++ti;
    } else {
      result[i] = outputs[i];
    }
  }

  return result;
}

std::unordered_map<int32_t, std::string> snapshotFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues) {
  std::unordered_map<int32_t, std::string> result;
  for (int32_t id = 0; id < numValues; ++id) {
    const auto& iv = frame.getIValue(id);
    if (iv.isNone()) {
      continue;
    }
    if (iv.isTensor()) {
      auto sizes = iv.toTensor().sizes();
      std::string shape;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) {
          shape += ",";
        }
        shape += std::to_string(sizes[i]);
      }
      result[id] = "[" + shape + "]";
    } else {
      result[id] = "scalar";
    }
  }
  return result;
}

void compareFrameSnapshots(
    const std::unordered_map<int32_t, std::string>& serial,
    const std::unordered_map<int32_t, std::string>& wave,
    int32_t numValues) {
  int32_t mismatches = 0;
  for (int32_t id = 0; id < numValues; ++id) {
    auto serialIt = serial.find(id);
    auto waveIt = wave.find(id);
    bool serialSet = serialIt != serial.end();
    bool waveSet = waveIt != wave.end();
    if (serialSet != waveSet) {
      LOG(WARNING) << "Frame mismatch at id " << id
                   << ": serial=" << (serialSet ? serialIt->second : "none")
                   << " wave=" << (waveSet ? waveIt->second : "none");
      ++mismatches;
    } else if (serialSet && serialIt->second != waveIt->second) {
      LOG(WARNING) << "Frame shape mismatch at id " << id
                   << ": serial=" << serialIt->second
                   << " wave=" << waveIt->second;
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    LOG(WARNING) << "Total frame mismatches: " << mismatches << " (serial has "
                 << serial.size() << " values, wave has " << wave.size()
                 << " values)";
  }
}

void standaloneReport(WaveGraphExecutor& executor) {
  auto stats = executor.getStandaloneStats();
  if (stats.empty()) {
    return;
  }
  std::sort(stats.begin(), stats.end(), [](const auto& a, const auto& b) {
    return a.second > b.second;
  });
  LOG(INFO) << "Standalone timing:";
  for (const auto& [name, micros] : stats) {
    LOG(INFO) << "  " << micros << " us  " << name;
  }
}

// --- ModelFixture ---

std::unique_ptr<ModelFixture> ModelFixture::load(const std::string& pt2Path) {
  auto fixture = std::make_unique<ModelFixture>();
  fixture->pt2Path = pt2Path;

  fixture->reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));

  auto modelNames = getModelNames(*fixture->reader);
  EXPECT_FALSE(modelNames.empty()) << "No models found in " << pt2Path;
  if (modelNames.empty()) {
    return nullptr;
  }

  fixture->model = loadPt2Model(fixture->reader, modelNames[0]);
  prepareGraph(fixture->model.graph.get());
  const auto& graph = *fixture->model.graph;

  fixture->weights = std::make_shared<nativert::Weights>(
      &graph,
      fixture->reader,
      fixture->model.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      fixture->model.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  return fixture;
}

void ModelFixture::prepareGraph(nativert::Graph* graph) {
  nativert::selectScalarOverload(graph);
}

std::vector<std::unique_ptr<nativert::OpKernel>> ModelFixture::makeKernels()
    const {
  nativert::KernelFactory factory;
  auto execKernels =
      factory.initializeNodeKernels(*model.graph, weights, config, reader);
  return std::move(execKernels.nodeKernels);
}

std::unique_ptr<ModelContext> ModelFixture::makeModelContext() {
  TORCH_CHECK(
      model.graph != nullptr,
      "ModelFixture::makeModelContext: graph already moved");
  auto ctx = std::make_unique<ModelContext>();
  ctx->graph = std::move(model.graph);
  ctx->weights = weights;
  ctx->config = config;
  return ctx;
}

// --- ExecutorTestBase ---

void ExecutorTestBase::SetUpTestSuite() {
  // Diagnostic: print device properties and test basic CUDA ops.
  LOG(INFO) << "CUDA device count: " << at::cuda::device_count();
  auto* props = at::cuda::getDeviceProperties(0);
  LOG(INFO) << "Device 0: " << props->name << " sm_" << props->major
            << props->minor;

  // Test empty (no kernel launch) vs zeros (requires fill kernel).
  LOG(INFO) << "Testing at::empty on CUDA...";
  try {
    auto e = at::empty(
        {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    LOG(INFO) << "at::empty OK, data_ptr=" << e.data_ptr();
  } catch (const std::exception& e) {
    LOG(ERROR) << "at::empty FAILED: " << e.what();
  }

  LOG(INFO) << "Testing at::ones on CUDA (fill kernel)...";
  try {
    auto o = at::ones(
        {4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    LOG(INFO) << "at::ones OK: " << o.cpu().sum().item<float>();
  } catch (const std::exception& e) {
    LOG(ERROR) << "at::ones FAILED: " << e.what();
  }

  WaveConfig::get().printTiming = FLAGS_print_timing;
  WaveConfig::get().allStandalone = FLAGS_standalone_kernels;
  WaveConfig::get().blockSize = FLAGS_block_dim;
  WaveConfig::get().trace = FLAGS_trace;
  if (FLAGS_single_block >= 0) {
    WaveConfig::get().useSingleBlock = FLAGS_single_block != 0;
  }
  if (FLAGS_cg >= 0) {
    WaveConfig::get().isCg = FLAGS_cg != 0;
  }
  WaveConfig::get().kernelCacheDir = FLAGS_kernel_cache_dir;
  WaveConfig::get().maxElementwiseVars = FLAGS_max_elementwise_vars;
  WaveConfig::get().outOfLineExprSize = FLAGS_out_of_line_expr_size;
  WaveConfig::get().traceValues = FLAGS_trace_values;
  WaveConfig::get().tensorPrintElementLimit = FLAGS_tensor_print_limit;
  WaveConfig::get().reverify = FLAGS_reverify;
  WaveConfig::get().throwOnError = FLAGS_throw_on_error;
  WaveConfig::get().noElementwiseFastPath = FLAGS_no_elementwise_fast_path;
  WaveConfig::get().continueAfterMismatch = FLAGS_continue_after_mismatch;
  WaveConfig::get().kernelDebugOutput = FLAGS_kernel_debug_output;
  WaveConfig::get().debugSingleOps = FLAGS_debug_single_ops;
  WaveConfig::get().autoAdjustCost = FLAGS_auto_adjust_cost;
  WaveConfig::get().enableReuse = FLAGS_enable_reuse;
  WaveConfig::get().freeIntermediates = FLAGS_free_intermediates;
  if (!FLAGS_print_options.empty()) {
    NodePrinter::setDefaults(
        NodePrinter::parsePrintOptions(FLAGS_print_options));
  }

  initialize();
}

std::vector<c10::IValue> ExecutorTestBase::executeSerialWithTrace(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels) {
  auto traceState = parseTraceValues(FLAGS_trace_values);

  // Trace user inputs already in the frame.
  for (const auto* value : graph.userInputs()) {
    if (value) {
      std::vector<nativert::ValueId> ids = {value->id()};
      traceFrameValues("serial input", ids, frame, traceState);
    }
  }

  // Values used as the 'self' (first) input of an index_put node are scratch
  // buffers (a clone or div result) that the wave rewrites to an in-place
  // tw.masked_put_, reusing that buffer. The serial run keeps them functional,
  // so their recorded value would never match the wave's reused buffer, and
  // they are dead after the index_put. Skip recording them to avoid false
  // positives.
  std::unordered_set<int64_t> inPlaceSelfIds;
  if (!FLAGS_save_reference_frame.empty()) {
    for (size_t scanIdx = 1; scanIdx + 1 < nodeKernels.size(); ++scanIdx) {
      auto* scanNode = nodeKernels[scanIdx]->node();
      std::string target(scanNode->target());
      if (target.find("index_put") != std::string::npos &&
          !scanNode->inputs().empty() && scanNode->inputs()[0].value) {
        inPlaceSelfIds.insert(scanNode->inputs()[0].value->id());
      }
    }
  }

  for (size_t nodeIdx = 1; nodeIdx + 1 < nodeKernels.size(); ++nodeIdx) {
    auto* node = nodeKernels[nodeIdx]->node();
    nodeKernels[nodeIdx]->compute(frame);
    for (const auto& input : node->inputs()) {
      std::vector<nativert::ValueId> ids = {input.value->id()};
      traceFrameValues("serial in", ids, frame, traceState);
    }
    for (auto* output : node->outputs()) {
      if (output) {
        // Capture a deep copy of each output the instant its node ran, so a
        // later in-place overwrite of its storage cannot corrupt the reference.
        // Copy straight to CPU: on a GPU nativert run this keeps the whole
        // reference frame off the GPU, so capturing every intermediate does not
        // exhaust GPU memory (a full-graph frame at large batch otherwise
        // OOMs). copy=true forces a fresh tensor even when the value is already
        // on CPU. Skip scratch self-args of index_put (see inPlaceSelfIds
        // above).
        if (!FLAGS_save_reference_frame.empty() &&
            !inPlaceSelfIds.count(output->id())) {
          const auto& iv = frame.getIValue(output->id());
          if (iv.isTensor() && iv.toTensor().numel() > 0) {
            capturedRefOutputs_[output->id()] = iv.toTensor().detach().to(
                at::kCPU, /*non_blocking=*/false, /*copy=*/true);
          }
        }
        std::vector<nativert::ValueId> ids = {output->id()};
        traceFrameValues("serial out", ids, frame, traceState);
      }
    }
  }
  return frame.tryMoveUserOutputs();
}

RunTiming ExecutorTestBase::runSerial(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected) {
  const auto& graph = *fixture.model.graph;

  auto serialConfig = fixture.config;
  if (!FLAGS_save_reference_frame.empty()) {
    serialConfig.tryFreeUnmanagedValuesAfterUse = false;
  }
  auto kernels = fixture.makeKernels();
  nativert::SerialGraphExecutor executor(
      graph, std::move(kernels), serialConfig);

  auto frame = std::make_unique<nativert::ExecutionFrame>(
      graph, *fixture.weights, serialConfig);

  auto inputs = loadSampleInputs(fixture);
  LOG(INFO) << "Serial: loaded " << inputs.size() << " inputs, graph expects "
            << graph.signature().userInputs().size();
  for (size_t i = 0; i < std::min<size_t>(inputs.size(), 5); ++i) {
    LOG(INFO) << "  input[" << i << "]: tag=" << inputs[i].tagKind()
              << " isTensor=" << inputs[i].isTensor()
              << " isObject=" << inputs[i].isObject()
              << " isTuple=" << inputs[i].isTuple()
              << " isList=" << inputs[i].isList();
  }

  // Snapshot frame after construction (weights filled, before user inputs).
  serialFrameSnapshot_ =
      snapshotFrame(*frame, static_cast<int32_t>(graph.numValues()));
  LOG(INFO) << "Serial frame before execute: " << serialFrameSnapshot_.size()
            << " non-none slots of " << graph.numValues();

  auto traceState = parseTraceValues(FLAGS_trace_values);
  bool traceSerial = !traceState.empty();
  if (traceSerial) {
    serialConfig.tryFreeUnmanagedValuesAfterUse = false;
  }

  fillFrameInputs(graph, *frame, std::move(inputs));

  auto start = Clock::now();
  std::vector<c10::IValue> outputs;
  // When saving a reference, run node-by-node so each output is captured at the
  // moment its node assigns it (see capturedRefOutputs_).
  bool nodeByNode = traceSerial || !FLAGS_save_reference_frame.empty();
  if (nodeByNode) {
    auto traceKernels = fixture.makeKernels();
    nativert::SerialGraphExecutor tempExecutor(
        graph, std::move(traceKernels), serialConfig);
    outputs =
        executeSerialWithTrace(graph, *frame, tempExecutor.stealKernels());
  } else {
    outputs = executor.executeWithPrefilledFrame(*frame);
  }
  auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       Clock::now() - start)
                       .count();

  // Re-snapshot after execute to capture user inputs filled by the executor.
  serialFrameSnapshot_ =
      snapshotFrame(*frame, static_cast<int32_t>(graph.numValues()));
  LOG(INFO) << "Serial frame after execute: " << serialFrameSnapshot_.size()
            << " non-none slots of " << graph.numValues();

  if (!FLAGS_save_reference_frame.empty()) {
    // Prefer the per-node CPU copies (capturedRefOutputs_) over the live frame,
    // so a GPU serial run never has to hold every intermediate on the device.
    saveReferenceFrame(
        *frame, graph, capturedRefOutputs_, FLAGS_save_reference_frame);
    LOG(INFO) << "Saved reference frame to " << FLAGS_save_reference_frame;
  }

  verifyOutputs(outputs, expected, "serial-pass1");

  // Run a second time with the same frame to test frame reuse.
  auto inputs2 = loadSampleInputs(fixture);
  fillFrameInputs(graph, *frame, std::move(inputs2));
  auto outputs2 = executor.executeWithPrefilledFrame(*frame);
  verifyOutputs(outputs2, expected, "serial-pass2");

  return {0, executeUs, {}};
}

RunTiming ExecutorTestBase::runSerialOnDevice(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected) {
  auto& graph = *fixture.model.graph;

  auto kernels = fixture.makeKernels();
  nativert::SerialGraphExecutor executor(
      graph, std::move(kernels), fixture.config);

  auto frame = std::make_unique<nativert::ExecutionFrame>(
      graph, *fixture.weights, fixture.config);

  auto inputs = loadSampleInputs(fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  // Set CUDA device so that at::kCUDA allocations target the right GPU.
  auto* waveDevice = facebook::velox::wave::currentDevice();
  if (!waveDevice) {
    waveDevice = facebook::velox::wave::getDevice();
  }
  at::cuda::set_device(static_cast<c10::DeviceIndex>(waveDevice->deviceId));

  fillFrameInputs(graph, *frame, std::move(deviceInputs));

  auto start = Clock::now();
  auto outputs = executor.executeWithPrefilledFrame(*frame);
  auto executeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       Clock::now() - start)
                       .count();

  // The CPU serial run (runSerial) saves the authoritative reference; the
  // GPU-serial path intentionally does not save, so it cannot overwrite it.
  auto hostOutputs = outputsToHost(outputs, "serial-gpu");
  verifyOutputs(hostOutputs, expected, "serial-gpu");
  return {dataMovUs, executeUs, {}};
}

void ExecutorTestBase::fillWaveFrame(
    const nativert::Graph& graph,
    nativert::ExecutionFrame& frame,
    const std::vector<c10::IValue>& deviceInputs) {
  const auto& userInputNames = graph.signature().userInputs();
  // Flatten to match user input count.
  std::vector<c10::IValue> flat;
  for (const auto& v : deviceInputs) {
    flattenIValue(v, flat);
  }
  while (flat.size() > userInputNames.size()) {
    std::vector<c10::IValue> next;
    bool changed = false;
    for (auto& v : flat) {
      auto before = next.size();
      flattenIValue(v, next);
      if (next.size() - before != 1) {
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
    flat = std::move(next);
  }
  size_t flatIdx = 0;
  for (const auto& name : userInputNames) {
    auto* value = graph.tryGetValue(name);
    EXPECT_NE(value, nullptr) << "No value for input " << name;
    if (value && flatIdx < flat.size()) {
      const auto& iv = flat[flatIdx++];
      if (!iv.isNone()) {
        if (iv.isTensor()) {
          frame.setIValue(value->id(), c10::IValue(iv.toTensor().clone()));
        } else {
          frame.setIValue(value->id(), iv);
        }
      }
    }
  }
}

RunTiming ExecutorTestBase::runWave(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& expected,
    const std::function<void(nativert::ExecutionFrame&)>& alterInputs) {
  std::unordered_map<int32_t, c10::IValue> refFrame;
  if (!FLAGS_reference_frame.empty()) {
    refFrame = loadReferenceFrame(FLAGS_reference_frame);
    WaveConfig::get().referenceFrame = &refFrame;
    LOG(INFO) << "Loaded reference frame with " << refFrame.size()
              << " values from " << FLAGS_reference_frame;
  }

  WaveGraphExecutor waveExec(fixture.makeModelContext());
  auto& graph = waveExec.graph();

  auto pooledFrame = waveExec.getFrame();
  EXPECT_NE(pooledFrame, nullptr);

  auto inputs = loadSampleInputs(fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  fillWaveFrame(graph, *pooledFrame, deviceInputs);

  if (alterInputs) {
    alterInputs(*pooledFrame);
  }

  if (!serialFrameSnapshot_.empty()) {
    auto waveSnapshot =
        snapshotFrame(*pooledFrame, static_cast<int32_t>(graph.numValues()));
    LOG(INFO) << "Wave frame after fillWaveFrame: " << waveSnapshot.size()
              << " non-none slots of " << graph.numValues();
    compareFrameSnapshots(
        serialFrameSnapshot_,
        waveSnapshot,
        static_cast<int32_t>(graph.numValues()));
  }

  auto waveStart = Clock::now();
  auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
  auto waveUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - waveStart)
                    .count();

  waveExec.returnFrame(std::move(pooledFrame));

  lastRefTensorsChecked_ = waveExec.numRefTensorsChecked();
  lastRefNodesChecked_ = waveExec.numRefNodesChecked();

  WaveConfig::get().referenceFrame = nullptr;

  auto hostOutputs = outputsToHost(waveOutputs, "wave");
  verifyOutputs(hostOutputs, expected, "wave");
  return {dataMovUs, waveUs, waveThreadInfo().debugInfo};
}

std::string ExecutorTestBase::runWaveExpectError(
    ModelFixture& fixture,
    const std::function<void(nativert::ExecutionFrame&)>& alterInputs) {
  bool savedThrow = WaveConfig::get().throwOnError;
  WaveConfig::get().throwOnError = false;
  // Empty 'expected' makes runWave skip output verification: an erroring run
  // has no meaningful reference to compare against.
  runWave(fixture, /*expected=*/{}, alterInputs);
  std::string errors = waveThreadInfo().errors;
  WaveConfig::get().throwOnError = savedThrow;
  return errors;
}

void ExecutorTestBase::runTest(
    const std::string& pt2File,
    const std::string& resultsFile,
    const std::string& label) {
  auto baseDir = dataDir();
  auto pt2Path =
      pt2File[0] == '/' ? pt2File : getDataFilePath(baseDir, pt2File);
  auto resultsPath = resultsFile[0] == '/'
      ? resultsFile
      : getDataFilePath(baseDir, resultsFile);
  auto inputsPath = pt2Path.substr(0, pt2Path.size() - 4) + "_inputs.pt";
  auto fixture = ModelFixture::load(pt2Path);
  ASSERT_NE(fixture, nullptr);

  // Check if an external inputs file exists.
  if (std::ifstream(inputsPath).good()) {
    LOG(INFO) << "Loading external inputs from " << inputsPath;
    fixture->externalInputsPath = inputsPath;
  }

  std::vector<c10::IValue> expected;
  try {
    expected = loadReferenceValues(resultsPath);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to load reference values from " << resultsPath
                 << ": " << e.what() << " — skipping result verification";
  }

  runTestWithFixture(
      std::move(fixture), expected, label.empty() ? pt2File : label);
}

void ExecutorTestBase::runTestWithFixture(
    std::unique_ptr<ModelFixture> fixture,
    const std::vector<c10::IValue>& expected,
    const std::string& label) {
  displayName_ = label;
  auto displayName = label;

  const int repeats = FLAGS_num_repeats;

  if (!FLAGS_wave_only) {
    // Serial CPU — runs before device placement so graph has CPU attributes.
    auto serialCold = runSerial(*fixture, expected);
    int64_t serialMin = INT64_MAX, serialMax = 0, serialSum = 0;
    std::vector<int64_t> serialTimes(repeats);
    for (int i = 0; i < repeats; ++i) {
      auto t = runSerial(*fixture, expected);
      serialTimes[i] = t.executeUs;
      serialMin = std::min(serialMin, t.executeUs);
      serialMax = std::max(serialMax, t.executeUs);
      serialSum += t.executeUs;
    }
    std::sort(serialTimes.begin(), serialTimes.end());
    auto serialP90 = serialTimes[repeats * 90 / 100];
    LOG(INFO) << displayName << " serial CPU (" << repeats
              << " repeats): min=" << serialMin
              << " avg=" << serialSum / repeats << " p90=" << serialP90
              << " max=" << serialMax << " us (cold=" << serialCold.executeUs
              << " us)";

    // Set graph device to CUDA once — used by serial GPU below and wave.
    setGraphDevice(fixture->model.graph.get(), true);

    // Serial GPU.
    auto deviceCold = runSerialOnDevice(*fixture, expected);
    int64_t deviceMin = INT64_MAX, deviceMax = 0, deviceSum = 0;
    std::vector<int64_t> deviceTimes(repeats);
    for (int i = 0; i < repeats; ++i) {
      auto t = runSerialOnDevice(*fixture, expected);
      deviceTimes[i] = t.executeUs;
      deviceMin = std::min(deviceMin, t.executeUs);
      deviceMax = std::max(deviceMax, t.executeUs);
      deviceSum += t.executeUs;
    }
    std::sort(deviceTimes.begin(), deviceTimes.end());
    auto deviceP90 = deviceTimes[repeats * 90 / 100];
    LOG(INFO) << displayName
              << " serial GPU H2D: cold=" << deviceCold.dataTransferUs << " us";
    LOG(INFO) << displayName << " serial GPU (" << repeats
              << " repeats): min=" << deviceMin
              << " avg=" << deviceSum / repeats << " p90=" << deviceP90
              << " max=" << deviceMax << " us (cold=" << deviceCold.executeUs
              << " us)";
  } else {
    // wave_only: apply placement here since the serial GPU path was skipped.
    setGraphDevice(fixture->model.graph.get(), true);
  }

  // Wave.
  WaveGraphExecutor waveExec(fixture->makeModelContext());
  auto& graph = waveExec.graph();

  if (FLAGS_list > 0) {
    auto mode = static_cast<Listing>(FLAGS_list - 1);
    std::cout << waveExec.waveGraph()->toString(mode) << std::endl;
  }

  auto inputs = loadSampleInputs(*fixture);
  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);

  if (FLAGS_single_ops != 1) {
    auto pooledFrame = waveExec.getFrame();
    ASSERT_NE(pooledFrame, nullptr);

    // Initial run: fill frame, execute, verify.
    fillWaveFrame(graph, *pooledFrame, deviceInputs);

    auto waveStart = Clock::now();
    auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
    auto waveColdUs = std::chrono::duration_cast<std::chrono::microseconds>(
                          Clock::now() - waveStart)
                          .count();

    auto hostOutputs = outputsToHost(waveOutputs, "wave");
    verifyOutputs(hostOutputs, expected, "wave");
    if (!FLAGS_throw_on_error) {
      auto& errors = waveThreadInfo().errors;
      if (!errors.empty()) {
        LOG(ERROR) << "Wave kernel errors:\n" << errors;
      }
    }

    // Repeated runs: refill inputs and re-execute only.
    int64_t waveMin = INT64_MAX, waveMax = 0, waveSum = 0;
    std::vector<int64_t> waveTimes(repeats);
    std::vector<std::vector<DebugInfo>> lastDebugInfo;
    for (int i = 0; i < repeats; ++i) {
      fillWaveFrame(graph, *pooledFrame, deviceInputs);

      auto start = Clock::now();
      waveExec.executeWithPrefilledFrame(*pooledFrame);
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - start)
                    .count();
      waveTimes[i] = us;
      waveMin = std::min(waveMin, us);
      waveMax = std::max(waveMax, us);
      waveSum += us;
      if (i == repeats - 1) {
        lastDebugInfo = waveThreadInfo().debugInfo;
      }
      if (WaveConfig::get().trace & WaveConfig::kTiming) {
        const auto& report = waveThreadInfo().perfReport;
        if (!report.empty()) {
          std::cout << "--- repeat " << i << " ---\n" << report << std::endl;
        }
      }
    }

    waveExec.returnFrame(std::move(pooledFrame));

    std::sort(waveTimes.begin(), waveTimes.end());
    auto waveP90 = waveTimes[repeats * 90 / 100];
    LOG(INFO) << displayName << " wave H2D: cold=" << dataMovUs << " us";
    LOG(INFO) << displayName << " wave (" << repeats
              << " repeats): min=" << waveMin << " avg=" << waveSum / repeats
              << " p90=" << waveP90 << " max=" << waveMax
              << " us (cold=" << waveColdUs << " us)";

    for (size_t li = 0; li < lastDebugInfo.size(); ++li) {
      const auto& blocks = lastDebugInfo[li];
      struct OpStats {
        int64_t count{0};
        int64_t minClocks{std::numeric_limits<int64_t>::max()};
        int64_t maxClocks{0};
        int64_t sumClocks{0};
      };
      std::map<int32_t, OpStats> opMap;
      for (const auto& b : blocks) {
        auto& s = opMap[b.op];
        s.count++;
        s.minClocks = std::min(s.minClocks, b.clocks);
        s.maxClocks = std::max(s.maxClocks, b.clocks);
        s.sumClocks += b.clocks;
      }
      std::vector<std::pair<int32_t, OpStats>> sorted(
          opMap.begin(), opMap.end());
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return (a.second.sumClocks / a.second.count) <
            (b.second.sumClocks / b.second.count);
      });
      std::stringstream ss;
      ss << displayName << " launch " << li << ": ";
      bool first = true;
      for (const auto& [op, s] : sorted) {
        if (!first) {
          ss << ", ";
        }
        first = false;
        ss << "op " << op << " count: " << s.count << " " << s.minClocks << "/"
           << (s.sumClocks / s.count) << "/" << s.maxClocks;
      }
      LOG(INFO) << ss.str();
    }

    if (FLAGS_standalone_timing) {
      standaloneReport(waveExec);
    }
  }

  if (FLAGS_single_ops != 0) {
    WaveConfig::get().debugSingleOps = true;
    auto debugLabel = displayName + " debug_single_ops";
    auto debugFrame = waveExec.getFrame();
    ASSERT_NE(debugFrame, nullptr);
    for (int run = 0; run < 2; ++run) {
      // The frame is reused across runs; clear the prior run's intermediates so
      // this run recomputes from the refilled inputs.  Otherwise stale outputs
      // make standalone ops (including in-place mutations) look
      // already-computed (see nodeOutputsComputed) and get skipped, yielding
      // stale results.
      if (run > 0) {
        debugFrame->clearNonPersistentValues();
      }
      fillWaveFrame(graph, *debugFrame, deviceInputs);
      auto outputs = waveExec.executeWithPrefilledFrame(*debugFrame);
      if (run == 1) {
        auto hostOutputs = outputsToHost(outputs, debugLabel);
        verifyOutputs(hostOutputs, expected, debugLabel);
        if (!FLAGS_throw_on_error) {
          auto& errors = waveThreadInfo().errors;
          if (!errors.empty()) {
            LOG(ERROR) << "Wave kernel errors (debug_single_ops):\n" << errors;
          }
        }
      }
    }
    waveExec.returnFrame(std::move(debugFrame));
    WaveConfig::get().debugSingleOps = false;
  }
}

std::vector<c10::IValue> ExecutorTestBase::loadSampleInputs(
    ModelFixture& fixture) {
  // If an external inputs file is available, load from it.
  if (!fixture.externalInputsPath.empty()) {
    auto data = readFile(fixture.externalInputsPath);
    auto ivalue = pickleLoadWithTypes(data);
    std::vector<c10::IValue> inputs;
    if (ivalue.isList()) {
      for (const auto& item : ivalue.toListRef()) {
        flattenIValue(item, inputs);
      }
    } else {
      flattenIValue(ivalue, inputs);
    }
    return inputs;
  }

  auto modelName = getModelNames(*fixture.reader)[0];
  std::string sampleInputsPath = fmt::format(
      torch::_export::archive_spec::SAMPLE_INPUTS_FILENAME_FORMAT, modelName);

  std::vector<c10::IValue> inputs;
  if (fixture.reader->hasRecord(sampleInputsPath)) {
    auto size = fixture.reader->getRecordSize(sampleInputsPath);
    std::vector<char> buffers(size);
    fixture.reader->getRecord(sampleInputsPath, buffers.data(), size);
    auto value = pickleLoadWithTypes(buffers);
    if (value.isTuple() && value.toTupleRef().elements().size() == 2) {
      const auto& argsVal = value.toTupleRef().elements().at(0);
      if (argsVal.isTuple()) {
        for (const auto& arg : argsVal.toTupleRef().elements()) {
          flattenIValue(arg, inputs);
        }
      }
      const auto& kwargsVal = value.toTupleRef().elements().at(1);
      if (kwargsVal.isTuple()) {
        for (const auto& kwarg : kwargsVal.toTupleRef().elements()) {
          flattenIValue(kwarg, inputs);
        }
      } else if (kwargsVal.isGenericDict()) {
        auto dict = kwargsVal.toGenericDict();
        for (const auto& entry : dict) {
          flattenIValue(entry.value(), inputs);
        }
      }
    }
  }
  EXPECT_FALSE(inputs.empty()) << "No sample inputs found in package";
  return inputs;
}

void ExecutorTestBase::verifyOutputs(
    const std::vector<c10::IValue>& outputs,
    const std::vector<c10::IValue>& expected,
    const std::string& label) {
  if (expected.empty()) {
    return;
  }
  auto fullLabel = displayName_.empty() ? label : displayName_ + " " + label;
  ASSERT_EQ(outputs.size(), expected.size())
      << fullLabel << ": output count mismatch";
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& actual = outputs[i];
    const auto& exp = expected[i];
    if (exp.isTensor()) {
      ASSERT_TRUE(actual.isTensor())
          << fullLabel << " output " << i << " expected tensor, got non-tensor";
      auto match = tensorsMatch(actual.toTensor(), exp.toTensor());
      auto limit = WaveConfig::get().tensorPrintElementLimit;
      EXPECT_TRUE(match) << fullLabel << " output " << i
                         << " differs from expected"
                         << "\n  "
                         << firstDifference(actual.toTensor(), exp.toTensor())
                         << "\n  expected: "
                         << tensorDebugString(exp.toTensor(), limit)
                         << "\n  actual:   "
                         << tensorDebugString(actual.toTensor(), limit);
      if (!match && FLAGS_print_full_mismatch) {
        LOG(INFO) << fullLabel << " output " << i << " expected:\n"
                  << tensorToString(exp.toTensor());
        LOG(INFO) << fullLabel << " output " << i << " actual:\n"
                  << tensorToString(actual.toTensor());
      }
    } else if (exp.isDouble()) {
      ASSERT_TRUE(actual.isDouble())
          << fullLabel << " output " << i << " expected double, got "
          << actual.tagKind();
      EXPECT_NEAR(actual.toDouble(), exp.toDouble(), 1e-5)
          << fullLabel << " output " << i << " scalar mismatch";
    } else if (exp.isInt()) {
      ASSERT_TRUE(actual.isInt()) << fullLabel << " output " << i
                                  << " expected int, got " << actual.tagKind();
      EXPECT_EQ(actual.toInt(), exp.toInt())
          << fullLabel << " output " << i << " scalar mismatch";
    } else if (exp.isBool()) {
      ASSERT_TRUE(actual.isBool())
          << fullLabel << " output " << i << " expected bool, got "
          << actual.tagKind();
      EXPECT_EQ(actual.toBool(), exp.toBool())
          << fullLabel << " output " << i << " scalar mismatch";
    } else if (exp.isString()) {
      // String outputs are constant keys (e.g. KJT feature names) baked into
      // the graph; compare them by value.
      ASSERT_TRUE(actual.isString())
          << fullLabel << " output " << i << " expected string, got "
          << actual.tagKind();
      EXPECT_EQ(actual.toStringRef(), exp.toStringRef())
          << fullLabel << " output " << i << " string mismatch";
    } else if (exp.isNone()) {
      // None outputs are structural (e.g. optional/absent leaves in the
      // out-spec); the actual must also be None.
      EXPECT_TRUE(actual.isNone())
          << fullLabel << " output " << i << " expected None, got "
          << actual.tagKind();
    } else {
      ADD_FAILURE() << fullLabel << " output " << i
                    << " unsupported expected type: " << exp.tagKind();
    }
  }
}

void saveModelArchive(
    caffe2::serialize::PyTorchStreamReader& reader,
    const std::string& outPath) {
  caffe2::serialize::PyTorchStreamWriter writer(outPath);
  for (const auto& name : reader.getAllRecords()) {
    // Drop the data sections (data/weights, data/constants,
    // data/sample_inputs): the real tensors may contain user data that must not
    // be checked in, and omitting them keeps the test archive small. Synthetic
    // weights and inputs are regenerated from the .spec by --run_synthetic.
    // Only the graph serialization (models/...) and archive metadata are kept.
    if (name.rfind("data/", 0) == 0) {
      continue;
    }
    auto [dataPtr, size] = reader.getRecord(name);
    writer.writeRecord(name, dataPtr.get(), size);
  }
  writer.writeEndOfFile();
}

std::unique_ptr<ModelFixture> ExecutorTestBase::loadSyntheticFixture(
    const std::string& pt2Path) {
  // Graph-only load: the saved archive has empty data sections, so weights are
  // not read from it -- they are generated from the spec and assigned by the
  // caller.
  auto fixture = std::make_unique<ModelFixture>();
  fixture->pt2Path = pt2Path;
  fixture->reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));
  auto modelNames = getModelNames(*fixture->reader);
  if (modelNames.empty()) {
    return nullptr;
  }
  fixture->model = loadPt2Model(fixture->reader, modelNames[0]);
  ModelFixture::prepareGraph(fixture->model.graph.get());
  return fixture;
}

void ExecutorTestBase::saveSyntheticModel(
    ModelFixture& fixture,
    const std::vector<c10::IValue>& inputs,
    const std::string& path) {
  makeDatasetSpec(
      *fixture.model.graph, *fixture.weights, inputs, path + ".spec");
  saveModelArchive(*fixture.reader, path + ".pt2");
  LOG(INFO) << "Saved synthetic model to " << path << ".pt2 and " << path
            << ".spec";
}

std::vector<c10::IValue> ExecutorTestBase::runNativertReferenceWithInputs(
    ModelFixture& fixture,
    std::vector<c10::IValue> inputs,
    const std::string& refFramePath) {
  auto& graph = *fixture.model.graph;
  // Match wave's graph before running: drop runtime data-validation asserts
  // that synthetic data cannot satisfy, apply the subclass rewrites (e.g.
  // merge_and_dedup -> _tw) so both use the same kernel, then place on GPU and
  // insert the cpuOnly copies the generic nativert executor needs. The weights
  // are already GPU-resident (see runSynthetic), so ops that mix weights with
  // device inputs stay on one device -- the piece a CPU-weight reference got
  // wrong (isin / _assert_tensor_metadata device mismatches).
  stripDataAsserts(graph);
  applySyntheticGraphRewrites(graph);
  setGraphDevice(&graph, true);
  rewriteGpuIncompatibleOps(graph);
  insertCpuOnlyCopies(graph);

  nativert::SerialGraphExecutor executor(
      graph, fixture.makeKernels(), fixture.config);
  auto frame = std::make_unique<nativert::ExecutionFrame>(
      graph, *fixture.weights, fixture.config);

  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);
  auto* waveDevice = facebook::velox::wave::currentDevice();
  if (!waveDevice) {
    waveDevice = facebook::velox::wave::getDevice();
  }
  at::cuda::set_device(static_cast<c10::DeviceIndex>(waveDevice->deviceId));

  fillFrameFromUserInputs(graph, *frame, deviceInputs);
  auto outputs = executor.executeWithPrefilledFrame(*frame);
  if (!refFramePath.empty()) {
    saveReferenceFrame(*frame, graph, refFramePath);
  }
  return outputsToHost(outputs, "synthetic-ref");
}

void ExecutorTestBase::runWaveWithInputs(
    ModelFixture& fixture,
    std::vector<c10::IValue> inputs,
    const std::vector<c10::IValue>& expected,
    const std::string& refFramePath) {
  std::unordered_map<int32_t, c10::IValue> refFrame;
  if (!refFramePath.empty()) {
    refFrame = loadReferenceFrame(refFramePath);
    WaveConfig::get().referenceFrame = &refFrame;
  }

  WaveGraphExecutor waveExec(fixture.makeModelContext());
  auto& graph = waveExec.graph();
  auto pooledFrame = waveExec.getFrame();

  auto [deviceInputs, dataMovUs] = inputsToDevice(inputs);
  fillFrameFromUserInputs(graph, *pooledFrame, deviceInputs);
  auto waveOutputs = waveExec.executeWithPrefilledFrame(*pooledFrame);
  waveExec.returnFrame(std::move(pooledFrame));

  lastRefTensorsChecked_ = waveExec.numRefTensorsChecked();
  lastRefNodesChecked_ = waveExec.numRefNodesChecked();
  WaveConfig::get().referenceFrame = nullptr;
  if (!refFramePath.empty()) {
    LOG(INFO) << "synthetic-wave reference frame: checked "
              << lastRefTensorsChecked_ << " tensors, " << lastRefNodesChecked_
              << " nodes";
  }

  auto hostOutputs = outputsToHost(waveOutputs, "synthetic-wave");

  // Compare wave outputs against the nativert reference the way the sigmoid
  // [refout] check does: tensors only, non-fatal, counting mismatches. Some
  // final outputs are legitimately None in wave (unmaterialized), so a strict
  // verifyOutputs would spuriously fail; the per-intermediate reference-frame
  // check above is the strict correctness gate.
  size_t numToCheck = std::min(hostOutputs.size(), expected.size());
  int compared = 0, mismatched = 0, skipped = 0;
  for (size_t i = 0; i < numToCheck; ++i) {
    if (!expected[i].isTensor() || !hostOutputs[i].isTensor()) {
      ++skipped;
      continue;
    }
    ++compared;
    auto actual = hostOutputs[i].toTensor();
    auto exp = expected[i].toTensor();
    if (actual.sizes() != exp.sizes() || !tensorsMatch(actual, exp)) {
      ++mismatched;
      LOG(ERROR) << "synthetic-wave output " << i
                 << " differs from reference: " << firstDifference(actual, exp);
    }
  }
  LOG(INFO) << "synthetic-wave: " << compared << " outputs compared, "
            << mismatched << " mismatched, " << skipped
            << " skipped (non-tensor)";
  EXPECT_EQ(mismatched, 0) << "synthetic-wave: " << mismatched
                           << " output(s) differ from the nativert reference";
}

void ExecutorTestBase::runSynthetic(
    const std::string& path,
    std::optional<uint64_t> seed) {
  std::string pt2Path = path + ".pt2";
  const std::string specPath = path + ".spec";

  // Committed graph archives are gzipped (<path>.pt2.gz) to stay under the repo
  // per-file size limit. When the plain .pt2 is absent, decompress to a temp
  // file and load that.
  std::string tempPt2;
  if (!std::filesystem::exists(pt2Path) &&
      std::filesystem::exists(pt2Path + ".gz")) {
    tempPt2 = decompressGzToTemp(pt2Path + ".gz");
    pt2Path = tempPt2;
  }

  // Weights are placed on the wave GPU so the GPU reference (and wave) run with
  // device-resident weights that match the device-placed graphs.
  auto* waveDevice = facebook::velox::wave::currentDevice();
  if (!waveDevice) {
    waveDevice = facebook::velox::wave::getDevice();
  }
  c10::Device weightDevice(
      c10::kCUDA, static_cast<c10::DeviceIndex>(waveDevice->deviceId));

  // Reference run: a fresh graph with generated data on the nativert GPU
  // executor produces the reference outputs and a reference frame.
  auto refFixture = loadSyntheticFixture(pt2Path);
  ASSERT_NE(refFixture, nullptr);
  auto generated =
      generateFromSpec(*refFixture->model.graph, specPath, seed, weightDevice);
  refFixture->weights = generated.weights;

  const std::string refFramePath =
      "/tmp/torchwave_synthetic_ref_" + std::to_string(getpid()) + ".pt";
  auto expected = runNativertReferenceWithInputs(
      *refFixture, generated.userInputs, refFramePath);

  // Wave run: another fresh graph, the same generated weights and inputs,
  // verified against the reference outputs and the reference frame.
  auto waveFixture = loadSyntheticFixture(pt2Path);
  ASSERT_NE(waveFixture, nullptr);
  waveFixture->weights = generated.weights;
  stripDataAsserts(*waveFixture->model.graph);
  applySyntheticGraphRewrites(*waveFixture->model.graph);
  setGraphDevice(waveFixture->model.graph.get(), true);
  runWaveWithInputs(*waveFixture, generated.userInputs, expected, refFramePath);

  std::remove(refFramePath.c_str());
  if (!tempPt2.empty()) {
    std::remove(tempPt2.c_str());
  }
}

} // namespace torch::wave
