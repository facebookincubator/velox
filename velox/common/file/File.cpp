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

#include "velox/common/file/File.h"
#ifdef VELOX_ENABLE_S3
#include "velox/common/file/S3File.h"
#endif
#include <fmt/format.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <fcntl.h>
#include <folly/portability/SysUio.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace facebook::velox {

namespace {

std::vector<std::function<void()>>& lazyRegistrations() {
  // Meyers singleton.
  static std::vector<std::function<void()>>* lazyRegs =
      new std::vector<std::function<void()>>();
  return *lazyRegs;
}
std::once_flag lazyRegistrationFlag;

using RegisteredReadFiles = std::vector<std::pair<
    std::function<bool(std::string_view)>,
    std::function<std::unique_ptr<
        ReadFile>(std::string_view, std::shared_ptr<const Config>)>>>;

RegisteredReadFiles& registeredReadFiles() {
  // Meyers singleton.
  static RegisteredReadFiles* files = new RegisteredReadFiles();
  return *files;
}

using RegisteredWriteFiles = std::vector<std::pair<
    std::function<bool(std::string_view)>,
    std::function<std::unique_ptr<
        WriteFile>(std::string_view, std::shared_ptr<const Config>)>>>;

RegisteredWriteFiles& registeredWriteFiles() {
  // Meyers singleton.
  static RegisteredWriteFiles* files = new RegisteredWriteFiles();
  return *files;
}

} // namespace

void lazyRegisterFileClass(std::function<void()> lazy_registration) {
  lazyRegistrations().push_back(lazy_registration);
}

void registerFileClass(
    std::function<bool(std::string_view)> filenameMatcher,
    std::function<std::unique_ptr<ReadFile>(
        std::string_view,
        std::shared_ptr<const Config>)> readGenerator,
    std::function<std::unique_ptr<WriteFile>(
        std::string_view,
        std::shared_ptr<const Config>)> writeGenerator) {
  registeredReadFiles().emplace_back(filenameMatcher, readGenerator);
  registeredWriteFiles().emplace_back(filenameMatcher, writeGenerator);
}

std::unique_ptr<ReadFile> generateReadFile(
    std::string_view filename,
    std::shared_ptr<const Config> properties) {
  std::call_once(lazyRegistrationFlag, []() {
    for (auto registration : lazyRegistrations()) {
      registration();
    }
  });
  const auto& files = registeredReadFiles();
  for (const auto& p : files) {
    if (p.first(filename)) {
      return p.second(filename, properties);
    }
  }
  throw std::runtime_error(fmt::format(
      "No registered read file matched with filename '{}'", filename));
}

std::unique_ptr<WriteFile> generateWriteFile(
    std::string_view filename,
    std::shared_ptr<const Config> properties) {
  std::call_once(lazyRegistrationFlag, []() {
    for (auto registration : lazyRegistrations()) {
      registration();
    }
  });
  const auto& files = registeredWriteFiles();
  for (const auto& p : files) {
    if (p.first(filename)) {
      return p.second(filename, properties);
    }
  }
  throw std::runtime_error(fmt::format(
      "No registered write file matched with filename '{}'", filename));
}

std::string_view InMemoryReadFile::pread(
    uint64_t offset,
    uint64_t length,
    Arena* /*unused_arena*/) const {
  bytesRead_ += length;
  return file_.substr(offset, length);
}

std::string_view
InMemoryReadFile::pread(uint64_t offset, uint64_t length, void* buf) const {
  bytesRead_ += length;
  memcpy(buf, file_.data() + offset, length);
  return {static_cast<char*>(buf), length};
}

std::string InMemoryReadFile::pread(uint64_t offset, uint64_t length) const {
  bytesRead_ += length;
  return std::string(file_.data() + offset, length);
}

uint64_t InMemoryReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) {
  uint64_t numRead = 0;
  if (offset >= file_.size()) {
    return 0;
  }
  for (auto& range : buffers) {
    auto copySize = std::min<size_t>(range.size(), file_.size() - offset);
    if (range.data()) {
      memcpy(range.data(), file_.data() + offset, copySize);
    }
    offset += copySize;
    numRead += copySize;
  }
  return numRead;
}

void InMemoryWriteFile::append(std::string_view data) {
  file_->append(data);
}

uint64_t InMemoryWriteFile::size() const {
  return file_->size();
}

LocalReadFile::LocalReadFile(std::string_view path) {
  std::unique_ptr<char[]> buf(new char[path.size() + 1]);
  buf[path.size()] = 0;
  memcpy(buf.get(), path.data(), path.size());
  fd_ = open(buf.get(), O_RDONLY);
  if (fd_ < 0) {
    throw std::runtime_error("open failure in LocalReadFile constructor.");
  }
}

void LocalReadFile::preadInternal(uint64_t offset, uint64_t length, char* pos)
    const {
  bytesRead_ += length;
  auto bytesRead = ::pread(fd_, pos, length, offset);
  if (bytesRead != length) {
    throw std::runtime_error("fread failure in LocalReadFile::PReadInternal.");
  }
}

std::string_view
LocalReadFile::pread(uint64_t offset, uint64_t length, Arena* arena) const {
  char* pos = arena->reserve(length);
  preadInternal(offset, length, pos);
  return {pos, length};
}

std::string_view
LocalReadFile::pread(uint64_t offset, uint64_t length, void* buf) const {
  preadInternal(offset, length, static_cast<char*>(buf));
  return {static_cast<char*>(buf), length};
}

std::string LocalReadFile::pread(uint64_t offset, uint64_t length) const {
  // TODO: use allocator that doesn't initialize memory?
  std::string result(length, 0);
  char* pos = result.data();
  preadInternal(offset, length, pos);
  return result;
}

uint64_t LocalReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) {
  static char droppedBytes[8 * 1024];
  std::vector<struct iovec> iovecs;
  iovecs.reserve(buffers.size());
  for (auto& range : buffers) {
    if (!range.data()) {
      auto skipSize = range.size();
      while (skipSize) {
        auto bytes = std::min<size_t>(sizeof(droppedBytes), skipSize);
        iovecs.push_back({droppedBytes, bytes});
        skipSize -= bytes;
      }
    } else {
      iovecs.push_back({range.data(), range.size()});
    }
  }
  return folly::preadv(fd_, iovecs.data(), iovecs.size(), offset);
}

uint64_t LocalReadFile::size() const {
  if (size_ != -1) {
    return size_;
  }
  const off_t rc = lseek(fd_, 0, SEEK_END);
  if (rc < 0) {
    throw std::runtime_error("fseek failure in LocalReadFile::size.");
  }
  size_ = rc;
  return size_;
}

uint64_t LocalReadFile::memoryUsage() const {
  // TODO: does FILE really not use any more memory? From the stdio.h
  // source code it looks like it has only a single integer? Probably
  // we need to go deeper and see how much system memory is being taken
  // by the file descriptor the integer refers to?
  return sizeof(FILE);
}

LocalWriteFile::LocalWriteFile(std::string_view path) {
  std::unique_ptr<char[]> buf(new char[path.size() + 1]);
  buf[path.size()] = 0;
  memcpy(buf.get(), path.data(), path.size());
  {
    FILE* exists = fopen(buf.get(), "rb");
    if (exists != nullptr) {
      throw std::runtime_error(fmt::format(
          "Failure in LocalWriteFile: path '{}' already exists.", path));
    }
  }
  file_ = fopen(buf.get(), "ab");
  if (file_ == nullptr) {
    throw std::runtime_error("fread failure in LocalWriteFile constructor.");
  }
}

LocalWriteFile::~LocalWriteFile() {
  if (file_) {
    const int fclose_ret = fclose(file_);
    if (fclose_ret != 0) {
      // We cannot throw an exception from the destructor. Warn instead.
      LOG(WARNING) << "fclose failure in LocalWriteFile destructor";
    }
  }
}

void LocalWriteFile::append(std::string_view data) {
  const uint64_t bytes_written = fwrite(data.data(), 1, data.size(), file_);
  if (bytes_written != data.size()) {
    throw std::runtime_error("fwrite failure in LocalWriteFile::append.");
  }
}

void LocalWriteFile::flush() {
  auto ret = fflush(file_);
  if (ret != 0) {
    throw std::runtime_error("fflush failed in LocalWriteFile::flush.");
  }
}

uint64_t LocalWriteFile::size() const {
  return ftell(file_);
}

// S3 support implementation.

inline Aws::String getAwsString(const std::string& s) {
  return Aws::String(s.begin(), s.end());
}

inline std::string_view fromAwsString(const Aws::String& s) {
  return {s.data(), s.length()};
}

void S3FileSystem::configureDefaultCredentialChain() {
  credentials_provider_ =
      std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
}
void S3FileSystem::configureAccessKey() {
  credentials_provider_ =
      std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(
          getAwsString((config_->get("hive.s3.aws-access-key")).value()),
          getAwsString((config_->get("hive.s3.aws-secret-key")).value()),
          getAwsString(""));
}

std::string S3FileSystem::getAccessKey() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetAWSAccessKeyId()));
}

std::string S3FileSystem::getSecretKey() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetAWSSecretKey()));
}

std::string S3FileSystem::getSessionToken() const {
  auto credentials = credentials_provider_->GetAWSCredentials();
  return std::string(fromAwsString(credentials.GetSessionToken()));
}

std::mutex aws_init_lock;
Aws::SDKOptions aws_options;
std::atomic<bool> aws_initialized(false);

void InitializeS3() {
  std::lock_guard<std::mutex> lock(aws_init_lock);
  if (!aws_initialized.load()) {
    aws_options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Fatal;
    Aws::InitAPI(aws_options);
    aws_initialized.store(true);
  }
}

void S3FileSystem::init() {
  if (!getRegion().empty()) {
    client_config_.region = getAwsString(getRegion());
  }
  client_config_.endpointOverride = getEndPoint();
  if (getScheme() == "http") {
    client_config_.scheme = Aws::Http::Scheme::HTTP;
  } else if (getScheme() == "https") {
    client_config_.scheme = Aws::Http::Scheme::HTTPS;
  } else {
    throw std::runtime_error(
        fmt::format("Invalid S3 connection scheme '", getScheme(), "'"));
  }

  // use virtual addressing for S3 on AWS (end point is empty)
  const bool use_virtual_addressing = getEndPoint().empty();
  configureAccessKey();
  client_ = std::make_shared<Aws::S3::S3Client>(
      credentials_provider_,
      client_config_,
      Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
      use_virtual_addressing);
}

void S3ReadFile::preadInternal(uint64_t offset, uint64_t length, char* pos)
    const {
  // Read the desired range of bytes
  Aws::S3::Model::GetObjectRequest req;
  Aws::S3::Model::GetObjectResult result;
  auto first_sep = path_.find_first_of(kSep);
  req.SetBucket(getAwsString(path_.substr(0, first_sep)));
  req.SetKey(getAwsString(path_.substr(first_sep + 1)));
  std::stringstream ss;
  ss << "bytes=" << offset << "-" << offset + length - 1;
  req.SetRange(getAwsString(ss.str()));
  // TODO: Avoid copy below by using  req.SetResponseStreamFactory();
  // Reference: ARROW-8692
  auto outcome = client_->GetObject(req);
  if (!outcome.IsSuccess()) {
    throw std::runtime_error("failure in S3ReadFile::size preadInternal.");
  }
  result = std::move(outcome).GetResultWithOwnership();
  auto& stream = result.GetBody();
  stream.read(reinterpret_cast<char*>(pos), length);
}

std::string_view
S3ReadFile::pread(uint64_t offset, uint64_t length, Arena* arena) const {
  char* pos = arena->reserve(length);
  preadInternal(offset, length, pos);
  return {pos, length};
}

std::string_view S3ReadFile::pread(uint64_t offset, uint64_t length, void* buf)
    const {
  preadInternal(offset, length, static_cast<char*>(buf));
  return {static_cast<char*>(buf), length};
}

std::string S3ReadFile::pread(uint64_t offset, uint64_t length) const {
  // TODO: use allocator that doesn't initialize memory?
  std::string result(length, 0);
  char* pos = result.data();
  preadInternal(offset, length, pos);
  return result;
}

uint64_t S3ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers) {
  throw std::runtime_error("failure in S3ReadFile::preadv.");
  return -1;
}

uint64_t S3ReadFile::size() const {
  throw std::runtime_error("failure in S3ReadFile::size.");
}

uint64_t S3ReadFile::memoryUsage() const {
  throw std::runtime_error("failure in S3ReadFile::memoryUsage.");
  return 0;
}
// Register the local and S3 Files.
namespace {

struct LocalFileRegistrar {
  LocalFileRegistrar() {
    lazyRegisterFileClass([this]() { this->ActuallyRegister(); });
  }

  void ActuallyRegister() {
    // Note: presto behavior is to prefix local paths with 'file:'.
    // Check for that prefix and prune to absolute regular paths as needed.
    std::function<bool(std::string_view)> filename_matcher =
        [](std::string_view filename) {
          return filename.find("/") == 0 || filename.find("file:") == 0;
        };
    std::function<std::unique_ptr<ReadFile>(
        std::string_view, std::shared_ptr<const Config>)>
        read_generator = [](std::string_view filename,
                            std::shared_ptr<const Config> properties) {
          if (filename.find("file:") == 0) {
            // TODO: Cache the FileSystems
            return LocalFileSystem(properties).openReadFile(filename.substr(5));
          } else {
            return LocalFileSystem(properties).openReadFile(filename);
          }
        };
    std::function<std::unique_ptr<WriteFile>(
        std::string_view, std::shared_ptr<const Config>)>
        write_generator = [](std::string_view filename,
                             std::shared_ptr<const Config> properties) {
          // TODO: Cache the FileSystems
          if (filename.find("file:") == 0) {
            return LocalFileSystem(properties)
                .openWriteFile(filename.substr(5));
          } else {
            return LocalFileSystem(properties).openWriteFile(filename);
          }
        };
    registerFileClass(filename_matcher, read_generator, write_generator);
  }
};

const LocalFileRegistrar localFileRegistrar;

struct S3FileRegistrar {
  S3FileRegistrar() {
    lazyRegisterFileClass([this]() { this->ActuallyRegister(); });
  }

  void ActuallyRegister() {
    // Note: presto behavior is to prefix local paths with 'file:'.
    // Check for that prefix and prune to absolute regular paths as needed.
    std::function<bool(std::string_view)> filename_matcher =
        [](std::string_view filename) { return isS3File(filename); };
    std::function<std::unique_ptr<ReadFile>(
        std::string_view, std::shared_ptr<const Config>)>
        read_generator = [](std::string_view filename,
                            std::shared_ptr<const Config> properties) {
          // TODO: Cache the FileSystem
          auto s3fs = S3FileSystem(properties);
          s3fs.init();
          return s3fs.openReadFile(filename);
        };
    std::function<std::unique_ptr<WriteFile>(
        std::string_view, std::shared_ptr<const Config>)>
        write_generator = [](std::string_view filename,
                             std::shared_ptr<const Config> properties) {
          // TODO: Cache the FileSystem
          return S3FileSystem(properties).openWriteFile(filename);
        };
    registerFileClass(filename_matcher, read_generator, write_generator);
  }
};

const S3FileRegistrar s3FileRegistrar;

} // namespace

} // namespace facebook::velox
