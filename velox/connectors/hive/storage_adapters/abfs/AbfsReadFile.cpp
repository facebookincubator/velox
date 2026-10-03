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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsReadFile.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsAsyncRuntime.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/connectors/hive/storage_adapters/abfs/AzureClientProviderFactories.h"
#include "velox/connectors/hive/storage_adapters/abfs/FollyHttpTransport.h"

#include <azure/core/url.hpp>
#include <folly/IPAddress.h>
#include <folly/fibers/FiberManagerInternal.h>

#include <algorithm>
#include <limits>
#include <optional>

namespace facebook::velox::filesystems {

namespace {

constexpr size_t kDiscardBufferSize = 262'144; // 256K

struct FiberTransportBinding {
  FollyHttpTransport* transport{nullptr};
};

class RuntimeHttpTransport final : public Azure::Core::Http::HttpTransport {
 public:
  std::unique_ptr<Azure::Core::Http::RawResponse> Send(
      Azure::Core::Http::Request& request,
      const Azure::Core::Context& context) override {
    auto* transport = folly::fibers::local<FiberTransportBinding>().transport;
    if (transport == nullptr) {
      throw std::logic_error(
          "ABFS runtime HTTP transport used outside an endpoint fiber");
    }
    return transport->Send(request, context);
  }
};

class ScopedFiberTransportBinding final {
 public:
  explicit ScopedFiberTransportBinding(FollyHttpTransport& transport)
      : binding_(folly::fibers::local<FiberTransportBinding>()),
        previous_(binding_.transport) {
    binding_.transport = &transport;
  }

  ~ScopedFiberTransportBinding() {
    binding_.transport = previous_;
  }

 private:
  FiberTransportBinding& binding_;
  FollyHttpTransport* previous_;
};

uint64_t logicalLength(const std::vector<folly::Range<char*>>& buffers) {
  uint64_t length{0};
  for (const auto& buffer : buffers) {
    VELOX_CHECK_LE(
        buffer.size(),
        std::numeric_limits<uint64_t>::max() - length,
        "ABFS read length overflow");
    length += buffer.size();
  }
  return length;
}

void readExactly(
    Azure::Core::IO::BodyStream& body,
    uint8_t* destination,
    size_t length,
    bool requireExactBody) {
  const auto bytesRead = body.ReadToCount(destination, length);
  if (requireExactBody && bytesRead != length) {
    throw std::runtime_error(
        "ABFS Blob response body ended before the requested range");
  }
}

uint64_t scatterRead(
    AzureBlobClient& client,
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    bool requireExactBody) {
  const auto length = logicalLength(buffers);
  if (length == 0) {
    return 0;
  }
  VELOX_USER_CHECK_LE(
      offset,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "ABFS read offset exceeds the Azure Blob range limit");
  VELOX_USER_CHECK_LE(
      length,
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "ABFS read length exceeds the Azure Blob range limit");

  Azure::Core::Http::HttpRange range;
  range.Offset = static_cast<int64_t>(offset);
  range.Length = static_cast<int64_t>(length);

  Azure::Storage::Blobs::DownloadBlobOptions options;
  options.Range = range;
  auto response = client.download(options);
  VELOX_CHECK_NOT_NULL(response.Value.BodyStream);

  std::vector<uint8_t> discardBuffer;
  for (const auto& buffer : buffers) {
    auto remaining = buffer.size();
    if (buffer.data() != nullptr) {
      readExactly(
          *response.Value.BodyStream,
          reinterpret_cast<uint8_t*>(buffer.data()),
          remaining,
          requireExactBody);
      continue;
    }

    const auto discardBufferSize = std::min(remaining, kDiscardBufferSize);
    if (discardBuffer.size() < discardBufferSize) {
      discardBuffer.resize(discardBufferSize);
    }

    while (remaining > 0) {
      const auto readSize = std::min(remaining, discardBuffer.size());
      readExactly(
          *response.Value.BodyStream,
          discardBuffer.data(),
          readSize,
          requireExactBody);
      remaining -= readSize;
    }
  }
  return length;
}

AbfsAsyncEndpointOptions endpointOptions(
    std::string_view clientUrl,
    size_t maxConnections) {
  const Azure::Core::Url url{std::string{clientUrl}};
  const auto& scheme = url.GetScheme();
  VELOX_USER_CHECK(
      scheme == "http" || scheme == "https",
      "Native ABFS async reads require an HTTP or HTTPS Blob endpoint");
  const auto port = static_cast<uint16_t>(
      url.GetPort() == 0 ? (scheme == "https" ? 443 : 80) : url.GetPort());

  AbfsAsyncEndpointOptions options;
  options.endpointKey = fmt::format("{}://{}:{}", scheme, url.GetHost(), port);
  options.port = port;
  options.channelEndpoint.serverName = url.GetHost();
  options.channelEndpoint.security = scheme == "https"
      ? AsyncChannelSecurity::kTls
      : AsyncChannelSecurity::kPlaintext;
  options.maxConnections = maxConnections;

  if (auto address = folly::IPAddress::tryFromString(url.GetHost())) {
    options.channelEndpoint.connectAddress =
        folly::SocketAddress(address.value(), port);
  } else {
    options.hostname = url.GetHost();
  }
  return options;
}

} // namespace

class AbfsReadFile::Impl : public std::enable_shared_from_this<Impl> {
  constexpr static uint64_t kNaturalReadSize = 4'194'304; // 4M
  constexpr static uint64_t kReadConcurrency = 8;

 public:
  explicit Impl(
      std::string_view path,
      const config::ConfigBase& config,
      std::shared_ptr<AbfsAsyncRuntime> asyncRuntime,
      size_t maxAsyncConnectionsPerEndpoint)
      : asyncRuntime_(std::move(asyncRuntime)) {
    auto abfsPath = std::make_shared<AbfsPath>(path);
    filePath_ = abfsPath->filePath();

    Azure::Storage::Blobs::BlobClientOptions fiberOptions;
    std::optional<AzureAsyncReadContext> asyncContext;
    if (asyncRuntime_ != nullptr) {
      fiberOptions.Transport.Transport =
          std::make_shared<RuntimeHttpTransport>();
      fiberOptions.Retry.RetryDelayCallback =
          [runtime = std::weak_ptr<AbfsAsyncRuntime>(asyncRuntime_)](
              std::chrono::milliseconds delay,
              const Azure::Core::Context& context) {
            context.ThrowIfCancelled();
            auto lockedRuntime = runtime.lock();
            if (lockedRuntime == nullptr) {
              throw std::runtime_error("ABFS async runtime is unavailable");
            }
            lockedRuntime->waitForRetryDelay(delay);
            context.ThrowIfCancelled();
          };
      fiberOptions.Retry.MaxRetries = 0;
      asyncContext.emplace(
          AzureAsyncReadContext{
              fiberOptions,
              asyncRuntime_->authService(),
          });
    }
    auto clients = AzureClientProviderFactories::getReadFileClients(
        abfsPath,
        config,
        asyncContext.has_value() ? &asyncContext.value() : nullptr);
    syncFileClient_ = std::move(clients.sync);
    fiberFileClient_ = std::move(clients.fiber);
    if (asyncRuntime_ != nullptr && fiberFileClient_ == nullptr) {
      VELOX_USER_FAIL(
          "Selected Azure client provider does not support native async reads "
          "for account '{}' with auth context '{}': {}",
          abfsPath->accountName(),
          clients.providerContext,
          clients.asyncUnsupportedReason);
    }
    if (fiberFileClient_ != nullptr) {
      endpointOptions_ = endpointOptions(
          fiberFileClient_->getUrl(), maxAsyncConnectionsPerEndpoint);
    }
  }

  void initialize(const FileOptions& options) {
    if (options.fileSize.has_value()) {
      VELOX_CHECK_GE(
          options.fileSize.value(), 0, "File size must be non-negative");
      length_ = options.fileSize.value();
    }

    if (length_ != -1) {
      return;
    }

    try {
      auto properties = syncFileClient_->getProperties();
      length_ = properties.Value.BlobSize;
    } catch (Azure::Storage::StorageException& e) {
      throwStorageExceptionWithOperationDetails("GetProperties", filePath_, e);
    }
    VELOX_CHECK_GE(length_, 0);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context) const {
    std::vector<folly::Range<char*>> buffers = {
        folly::Range<char*>(static_cast<char*>(buffer), length)};
    preadv(offset, buffers, context);
    return {static_cast<char*>(buffer), length};
  }

  std::string
  pread(uint64_t offset, uint64_t length, const FileIoContext& context) const {
    std::string result(length, 0);
    std::vector<folly::Range<char*>> buffers = {
        folly::Range<char*>(result.data(), result.size())};
    preadv(offset, buffers, context);
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context) const {
    if (hasPreadvAsync()) {
      VELOX_CHECK(
          !asyncRuntime_->isRuntimeThread(),
          "Synchronous ABFS reads cannot run on an async runtime thread");
      return std::move(preadvAsync(offset, buffers, context)).get();
    }
    return scatterRead(*syncFileClient_, offset, buffers, false);
  }

  uint64_t preadv(
      folly::Range<const common::Region*> regions,
      folly::Range<folly::IOBuf*> iobufs,
      const FileIoContext& context) const {
    size_t length = 0;
    VELOX_CHECK_EQ(regions.size(), iobufs.size());
    for (size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      auto& output = iobufs[i];
      output = folly::IOBuf(folly::IOBuf::CREATE, region.length);
      pread(region.offset, region.length, output.writableData(), context);
      output.append(region.length);
      length += region.length;
    }

    return length;
  }

  uint64_t size() const {
    return length_;
  }

  uint64_t memoryUsage() const {
    return 3 * sizeof(std::string) + sizeof(int64_t);
  }

  bool shouldCoalesce() const {
    return false;
  }

  std::string getName() const {
    return filePath_;
  }

  uint64_t getNaturalReadSize() const {
    return kNaturalReadSize;
  }

  bool hasPreadvAsync() const {
    return asyncRuntime_ != nullptr && fiberFileClient_ != nullptr;
  }

  folly::SemiFuture<uint64_t> preadvAsync(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context) const {
    uint64_t length{0};
    try {
      length = logicalLength(buffers);
    } catch (...) {
      auto contract = folly::makePromiseContract<uint64_t>();
      contract.promise.setException(
          folly::exception_wrapper(std::current_exception()));
      return std::move(contract.future);
    }
    if (length == 0) {
      return folly::makeSemiFuture<uint64_t>(0);
    }

    auto request = asyncRuntime_->submit(
        endpointOptions_,
        [impl = shared_from_this(), offset, buffers, context](
            FollyHttpTransport& transport,
            const folly::CancellationToken&) mutable {
          ScopedFiberTransportBinding binding(transport);
          static_cast<void>(context);
          scatterRead(*impl->fiberFileClient_, offset, buffers, true);
        });
    return std::move(request).deferValue(
        [length](folly::Unit) { return length; });
  }

 private:
  std::string filePath_;
  std::unique_ptr<AzureBlobClient> syncFileClient_;
  std::unique_ptr<AzureBlobClient> fiberFileClient_;
  std::shared_ptr<AbfsAsyncRuntime> asyncRuntime_;
  AbfsAsyncEndpointOptions endpointOptions_;
  int64_t length_ = -1;
};

AbfsReadFile::AbfsReadFile(
    std::string_view path,
    const config::ConfigBase& config,
    std::shared_ptr<AbfsAsyncRuntime> asyncRuntime,
    size_t maxAsyncConnectionsPerEndpoint) {
  impl_ = std::make_shared<Impl>(
      path, config, std::move(asyncRuntime), maxAsyncConnectionsPerEndpoint);
}

void AbfsReadFile::initialize(const FileOptions& options) {
  impl_->initialize(options);
}

std::string_view AbfsReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buffer,
    const FileIoContext& context) const {
  return impl_->pread(offset, length, buffer, context);
}

std::string AbfsReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  return impl_->pread(offset, length, context);
}

uint64_t AbfsReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  return impl_->preadv(offset, buffers, context);
}

uint64_t AbfsReadFile::preadv(
    folly::Range<const common::Region*> regions,
    folly::Range<folly::IOBuf*> iobufs,
    const FileIoContext& context) const {
  return impl_->preadv(regions, iobufs, context);
}

bool AbfsReadFile::hasPreadvAsync() const {
  return impl_->hasPreadvAsync();
}

folly::SemiFuture<uint64_t> AbfsReadFile::preadvAsync(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  if (!impl_->hasPreadvAsync()) {
    return ReadFile::preadvAsync(offset, buffers, context);
  }
  return impl_->preadvAsync(offset, buffers, context);
}

uint64_t AbfsReadFile::size() const {
  return impl_->size();
}

uint64_t AbfsReadFile::memoryUsage() const {
  return impl_->memoryUsage();
}

bool AbfsReadFile::shouldCoalesce() const {
  return false;
}

std::string AbfsReadFile::getName() const {
  return impl_->getName();
}

uint64_t AbfsReadFile::getNaturalReadSize() const {
  return impl_->getNaturalReadSize();
}

} // namespace facebook::velox::filesystems
