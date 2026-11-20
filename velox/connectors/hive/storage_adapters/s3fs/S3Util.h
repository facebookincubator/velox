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

// Implementation of S3 filesystem and file interface.
// We provide a registration method for read and write files so the appropriate
// type of file can be constructed based on a filename. See the
// (register|generate)ReadFile and (register|generate)WriteFile functions.

#pragma once

#include <aws/s3/S3Errors.h>
#include <aws/s3/model/HeadObjectResult.h>
#include <fmt/format.h>
#include <folly/Uri.h>

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"

#include <aws/core/utils/stream/PreallocatedStreamBuf.h>

#include <algorithm>
#include <cstring>
#include <iostream>

namespace facebook::velox::filesystems {

namespace {
static std::string_view kSep{"/"};
// AWS S3 EMRFS, Hadoop block storage filesystem on-top of Amazon S3 buckets.
static std::string_view kS3Scheme{"s3://"};
// This should not be mixed with s3 nor the s3a.
// S3A Hadoop 3.x (previous connectors "s3" and "s3n" are deprecated).
static std::string_view kS3aScheme{"s3a://"};
// DEPRECATED: s3n are deprecated in Hadoop 3.x but we are supporting s3n for
// data that hasn't been migrated yet.
static std::string_view kS3nScheme{"s3n://"};
// OSS Alibaba support S3 format, usage only with SSL.
static std::string_view kOssScheme{"oss://"};
// Tencent COS support S3 format.
static std::string_view kCosScheme{"cos://"};
static std::string_view kCosNScheme{"cosn://"};

// From AWS documentation
constexpr int kS3MaxKeySize{1024};
} // namespace

inline bool isS3AwsFile(const std::string_view filename) {
  return filename.substr(0, kS3Scheme.size()) == kS3Scheme;
}

inline bool isS3aFile(const std::string_view filename) {
  return filename.substr(0, kS3aScheme.size()) == kS3aScheme;
}

inline bool isS3nFile(const std::string_view filename) {
  return filename.substr(0, kS3nScheme.size()) == kS3nScheme;
}

inline bool isOssFile(const std::string_view filename) {
  return filename.substr(0, kOssScheme.size()) == kOssScheme;
}

inline bool isCosFile(const std::string_view filename) {
  return filename.substr(0, kCosScheme.size()) == kCosScheme;
}

inline bool isCosNFile(const std::string_view filename) {
  return filename.substr(0, kCosNScheme.size()) == kCosNScheme;
}

inline bool isS3File(const std::string_view filename) {
  // TODO: Each prefix should be implemented as its own filesystem.
  return isS3AwsFile(filename) || isS3aFile(filename) || isS3nFile(filename) ||
      isOssFile(filename) || isCosFile(filename) || isCosNFile(filename);
}

// The input `path` must not have the S3 prefix.
inline void getBucketAndKeyFromPath(
    std::string_view path,
    std::string& bucket,
    std::string& key) {
  auto firstSep = path.find_first_of(kSep);
  bucket = path.substr(0, firstSep);
  key = path.substr(firstSep + 1);
}

// TODO: Correctness check for bucket name.
// 1. Length between 3 and 63:
//    3 < length(bucket) < 63
// 2. Mandatory label notation - regexp:
//    regexp="(^[a-z0-9])([.-]?[a-z0-9]+){2,62}([/]?$)"
// 3. Disallowed IPv4 notation - regexp:
//    regexp="^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}[/]?$"
inline std::string s3URI(std::string_view bucket, std::string_view key) {
  std::stringstream ss;
  ss << kS3Scheme << bucket << kSep << key;
  return ss.str();
}

inline std::string_view getPath(std::string_view path) {
  // Remove one of the prefixes 's3://', 'oss://', 's3a://' if any from the
  // given path.
  // TODO: Each prefix should be implemented as its own filesystem.
  if (isS3AwsFile(path)) {
    return path.substr(kS3Scheme.length());
  } else if (isS3aFile(path)) {
    return path.substr(kS3aScheme.length());
  } else if (isS3nFile(path)) {
    return path.substr(kS3nScheme.length());
  } else if (isOssFile(path)) {
    return path.substr(kOssScheme.length());
  } else if (isCosFile(path)) {
    return path.substr(kCosScheme.length());
  } else if (isCosNFile(path)) {
    return path.substr(kCosNScheme.length());
  }
  return path;
}

inline Aws::String awsString(const std::string& s) {
  return Aws::String(s.begin(), s.end());
}

std::string getErrorStringFromS3Error(
    const Aws::Client::AWSError<Aws::S3::S3Errors>& error);

namespace {
inline std::string getS3BackendService(
    const Aws::Http::HeaderValueCollection& headers) {
  const auto it = headers.find("server");
  if (it != headers.end()) {
    return it->second;
  }
  return "Unknown";
}

inline std::string getRequestID(
    const Aws::Http::HeaderValueCollection& headers) {
  static const std::vector<std::string> kRequestIds{
      "x-amz-request-id", "x-oss-request-id"};

  for (const auto& kRequestId : kRequestIds) {
    const auto it = headers.find(kRequestId);
    if (it != headers.end()) {
      return it->second;
    }
  }
  return "";
}
} // namespace

/// Only Amazon (amz) and Alibaba (oss) request IDs are supported.
#define VELOX_CHECK_AWS_OUTCOME(outcome, errorMsgPrefix, bucket, key)                                                           \
  {                                                                                                                             \
    if (!outcome.IsSuccess()) {                                                                                                 \
      auto error = outcome.GetError();                                                                                          \
      auto errMsg = fmt::format(                                                                                                \
          "{} due to: '{}'. Path:'{}', SDK Error Type:{}, HTTP Status Code:{}, S3 Service:'{}', Message:'{}', RequestID:'{}'.", \
          errorMsgPrefix,                                                                                                       \
          getErrorStringFromS3Error(error),                                                                                     \
          s3URI(bucket, key),                                                                                                   \
          static_cast<int>(error.GetErrorType()),                                                                               \
          error.GetResponseCode(),                                                                                              \
          getS3BackendService(error.GetResponseHeaders()),                                                                      \
          error.GetMessage(),                                                                                                   \
          getRequestID(error.GetResponseHeaders()));                                                                            \
      if (IsRetryableHttpResponseCode(error.GetResponseCode()) ||                                                               \
          error.ShouldRetry()) {                                                                                                \
        errMsg.append(                                                                                                          \
            fmt::format(                                                                                                        \
                " Request failed after retrying {} times. Try increasing the value of the S3 'max-attempts' configuration.",    \
                outcome.GetRetryCount()));                                                                                      \
      }                                                                                                                         \
      if (error.GetResponseCode() == Aws::Http::HttpResponseCode::NOT_FOUND) {                                                  \
        VELOX_FILE_NOT_FOUND_ERROR(errMsg);                                                                                     \
      }                                                                                                                         \
      VELOX_FAIL(errMsg);                                                                                                       \
    }                                                                                                                           \
  }

bool isHostExcludedFromProxy(
    const std::string& hostname,
    const std::string& noProxyList);

std::string getHttpProxyEnvVar();
std::string getHttpsProxyEnvVar();
std::string getNoProxyEnvVar();

// Adopted from the AWS Java SDK
// Endpoint can be 'service.[region].amazonaws.com' or
// 'bucket.s3-[region].amazonaws.com' or bucket.s3.[region].amazonaws.com'
// Return value is a region string value if present.
// The endpoint may contain a trailing '/' that is handled.
std::optional<std::string> parseAWSStandardRegionName(
    std::string_view endpoint);

/// Returns whether to use AWS FIPS endpoints: 's3.use-fips' when set, else the
/// AWS_USE_FIPS_ENDPOINT environment variable, which the AWS C++ SDK does not
/// read itself. False when neither is set.
bool useFipsEndpoint(const S3Config& s3Config);

class S3ProxyConfigurationBuilder {
 public:
  S3ProxyConfigurationBuilder(const std::string& s3Endpoint)
      : s3Endpoint_(s3Endpoint) {}

  S3ProxyConfigurationBuilder& useSsl(const bool& useSsl) {
    useSsl_ = useSsl;
    return *this;
  }

  std::optional<folly::Uri> build();

 private:
  const std::string s3Endpoint_;
  bool useSsl_;
};

// Reference: https://issues.apache.org/jira/browse/ARROW-8692
// https://github.com/apache/arrow/blob/master/cpp/src/arrow/filesystem/s3fs.cc#L843
// A non-copying iostream. See
// https://stackoverflow.com/questions/35322033/aws-c-sdk-uploadpart-times-out
// https://stackoverflow.com/questions/13059091/creating-an-input-stream-from-constant-memory
class StringViewStream : Aws::Utils::Stream::PreallocatedStreamBuf,
                         public std::iostream {
 public:
  StringViewStream(const void* data, int64_t nbytes)
      : Aws::Utils::Stream::PreallocatedStreamBuf(
            reinterpret_cast<unsigned char*>(const_cast<void*>(data)),
            static_cast<size_t>(nbytes)),
        std::iostream(this) {}
};

/// A stream over a caller-provided buffer for the AWS SDK to write a response
/// body into. Bytes written past the buffer are counted and dropped instead of
/// failing the stream. The SDK writes an error response, such as the XML body
/// of a 503 SlowDown, into the same stream as object data; a stream that fails
/// on it turns the response into a non-retryable INTERNAL_FAILURE ("Failed to
/// flush response stream"), while one that stays good lets the SDK classify
/// the response by its HTTP status and retry it. Reads return the body when it
/// fit in the buffer, which is how the SDK parses an error body, and nothing
/// when it did not: the SDK's XML parser can crash on a cut-off body, and the
/// HTTP status alone classifies the response.
class ResponseBufferStream : private std::streambuf, public std::iostream {
 public:
  // Both bases define these.
  using int_type = std::streambuf::int_type;
  using off_type = std::streambuf::off_type;
  using pos_type = std::streambuf::pos_type;
  using traits_type = std::streambuf::traits_type;

  ResponseBufferStream(char* data, size_t capacity)
      : std::iostream(this), data_(data), capacity_(capacity) {
    setp(data_, data_ + capacity_);
    setg(data_, data_, data_);
  }

  /// Bytes written by the SDK, including those dropped past the buffer.
  size_t bytesReceived() const {
    return received_;
  }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    const auto kept = std::min<std::streamsize>(n, epptr() - pptr());
    std::memcpy(pptr(), s, kept);
    // setp() rather than pbump(), whose offset is an int.
    setp(pptr() + kept, epptr());
    received_ += n;
    return n;
  }

  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    if (pptr() < epptr()) {
      *pptr() = traits_type::to_char_type(ch);
      pbump(1);
    }
    ++received_;
    return ch;
  }

  int_type underflow() override {
    char* const end = readableEnd();
    setg(data_, std::min(gptr(), end), end);
    return gptr() < egptr() ? traits_type::to_int_type(*gptr())
                            : traits_type::eof();
  }

  pos_type seekoff(
      off_type off,
      std::ios_base::seekdir dir,
      std::ios_base::openmode which) override {
    const bool in = which & std::ios_base::in;
    const bool out = which & std::ios_base::out;
    if (in == out) {
      return pos_type(off_type(-1));
    }
    const off_type end =
        in ? readableEnd() - data_ : static_cast<off_type>(capacity_);
    off_type base = 0;
    if (dir == std::ios_base::cur) {
      base = (in ? gptr() : pptr()) - data_;
    } else if (dir == std::ios_base::end) {
      base = end;
    }
    const off_type pos = base + off;
    if (pos < 0 || pos > end) {
      return pos_type(off_type(-1));
    }
    if (in) {
      setg(data_, data_ + pos, data_ + end);
    } else {
      setp(data_ + pos, data_ + capacity_);
    }
    return pos_type(pos);
  }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(off_type(pos), std::ios_base::beg, which);
  }

 private:
  // The end of the readable bytes: those written, or none once the body
  // overran the buffer.
  char* readableEnd() const {
    return received_ > capacity_ ? data_ : pptr();
  }

  char* const data_;
  const size_t capacity_;
  size_t received_{0};
};

} // namespace facebook::velox::filesystems

template <>
struct fmt::formatter<Aws::Http::HttpResponseCode> : formatter<int> {
  auto format(Aws::Http::HttpResponseCode s, format_context& ctx) const {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};
