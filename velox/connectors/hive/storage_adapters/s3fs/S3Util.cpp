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

#include "folly/IPAddress.h"
#include "re2/re2.h"

#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

namespace facebook::velox::filesystems {

std::string getErrorStringFromS3Error(
    const Aws::Client::AWSError<Aws::S3::S3Errors>& error) {
  switch (error.GetErrorType()) {
    case Aws::S3::S3Errors::NO_SUCH_BUCKET:
      return "No such bucket";
    case Aws::S3::S3Errors::NO_SUCH_KEY:
      return "No such key";
    case Aws::S3::S3Errors::RESOURCE_NOT_FOUND:
      return "Resource not found";
    case Aws::S3::S3Errors::ACCESS_DENIED:
      return "Access denied";
    case Aws::S3::S3Errors::SERVICE_UNAVAILABLE:
      return "Service unavailable";
    case Aws::S3::S3Errors::NETWORK_CONNECTION:
      return "Network connection";
    case Aws::S3::S3Errors::INTERNAL_FAILURE:
      return "Internal failure";
    case Aws::S3::S3Errors::THROTTLING:
      return "Throttling";
    case Aws::S3::S3Errors::SLOW_DOWN:
      return "Slow down";
    case Aws::S3::S3Errors::REQUEST_TIMEOUT:
      return "Request timeout";
    case Aws::S3::S3Errors::REQUEST_EXPIRED:
      return "Request expired";
    case Aws::S3::S3Errors::REQUEST_TIME_TOO_SKEWED:
      return "Request time too skewed";
    case Aws::S3::S3Errors::INVALID_ACCESS_KEY_ID:
      return "Invalid access key ID";
    case Aws::S3::S3Errors::INVALID_OBJECT_STATE:
      return "Invalid object state";
    default:
      const auto& exceptionName = error.GetExceptionName();
      if (!exceptionName.empty()) {
        return std::string(exceptionName.c_str());
      }
      return "Unknown error";
  }
}

/// The noProxyList is a comma separated list of subdomains, domains or IP
/// ranges (using CIDR). For a given hostname check if it has a matching
/// subdomain, domain or IP range in the noProxyList.
bool isHostExcludedFromProxy(
    const std::string& hostname,
    const std::string& noProxyList) {
  std::vector<std::string> noProxyListElements{};

  if (noProxyList.empty()) {
    return false;
  }

  auto hostAsIp = folly::IPAddress::tryFromString(hostname);
  folly::split(',', noProxyList, noProxyListElements);
  for (auto elem : noProxyListElements) {
    // Elem contains "/" which separates IP and subnet mask e.g. 192.168.1.0/24.
    if (elem.find("/") != std::string::npos && hostAsIp.hasValue()) {
      return hostAsIp.value().inSubnet(elem);
    }
    // Match subdomain, domain names and IP address strings.
    else if (
        elem.length() < hostname.length() && elem[0] == '.' &&
        !hostname.compare(
            hostname.length() - elem.length(), elem.length(), elem)) {
      return true;
    } else if (
        elem.length() < hostname.length() && elem[0] == '*' && elem[1] == '.' &&
        !hostname.compare(
            hostname.length() - elem.length() + 1,
            elem.length() - 1,
            elem.substr(1))) {
      return true;
    } else if (elem.length() == hostname.length() && !hostname.compare(elem)) {
      return true;
    }
  }
  return false;
}

/// Reading the various proxy related environment variables.
/// There is a lacking standard. The environment variables can be
/// defined lower case or upper case. The lower case values are checked
/// first and, if set, returned, therefore taking precendence.
/// Note, the envVar input is expected to be lower case.
namespace {
std::string readProxyEnvVar(std::string envVar) {
  auto httpProxy = getenv(envVar.c_str());
  if (httpProxy) {
    return std::string(httpProxy);
  }

  std::transform(envVar.begin(), envVar.end(), envVar.begin(), ::toupper);
  httpProxy = getenv(envVar.c_str());
  if (httpProxy) {
    return std::string(httpProxy);
  }
  return "";
};
} // namespace

std::string getHttpProxyEnvVar() {
  return readProxyEnvVar("http_proxy");
}

std::string getHttpsProxyEnvVar() {
  return readProxyEnvVar("https_proxy");
};

std::string getNoProxyEnvVar() {
  return readProxyEnvVar("no_proxy");
};

std::optional<folly::Uri> S3ProxyConfigurationBuilder::build() {
  std::string proxyUrl;
  if (useSsl_) {
    proxyUrl = getHttpsProxyEnvVar();
  } else {
    proxyUrl = getHttpProxyEnvVar();
  }

  if (proxyUrl.empty()) {
    return std::nullopt;
  }
  folly::Uri proxyUri(proxyUrl);

  /// The endpoint is usually a domain with port or an
  /// IP address with port. It is assumed that there are
  /// 2 parts separated by a colon.
  std::vector<std::string> endpointElements{};
  folly::split(':', s3Endpoint_, endpointElements);
  if (FOLLY_UNLIKELY(endpointElements.size() > 2)) {
    LOG(ERROR) << fmt::format(
        "Too many parts in S3 endpoint URI {} ", s3Endpoint_);
    return std::nullopt;
  }

  auto noProxy = getNoProxyEnvVar();
  if (isHostExcludedFromProxy(endpointElements[0], noProxy)) {
    return std::nullopt;
  }
  return proxyUri;
}

namespace {
// The assumption is that an AWS endpoint ends with ".amazonaws.com" or
// ".amazonaws.com/". That means for AWS we don't expect a port in the endpoint.
const std::string_view kAmazonHostSuffix{".amazonaws.com"};

std::string_view withoutTrailingSlash(std::string_view endpoint) {
  if (!endpoint.empty() && endpoint.back() == '/') {
    endpoint.remove_suffix(1);
  }
  return endpoint;
}
} // namespace

bool isAWSEndpoint(std::string_view endpoint) {
  endpoint = withoutTrailingSlash(endpoint);
  // A shorter endpoint underflows the subtraction below to npos, which rfind
  // also returns when the suffix is absent, so the match would pass.
  if (endpoint.size() <= kAmazonHostSuffix.size()) {
    return false;
  }
  return endpoint.rfind(kAmazonHostSuffix) ==
      endpoint.size() - kAmazonHostSuffix.size();
}

std::optional<std::string> defaultRegionForEndpoint(std::string_view endpoint) {
  if (endpoint.empty()) {
    // No endpoint means AWS. Returning a region keeps the SDK from silently
    // picking us-east-1, or the EC2 instance's own region, for a bucket that
    // may live elsewhere.
    return std::string(kS3AwsGlobalRegion);
  }
  auto region = parseAWSStandardRegionName(endpoint);
  if (region.has_value()) {
    return region;
  }
  // An AWS endpoint whose host names no region, e.g. 's3.amazonaws.com'. For
  // anything else return nullopt and leave the region to the SDK, which honors
  // AWS_REGION, the active profile, and IMDS. Note that aws-imds-enabled cannot
  // stand in for this test: it is a permission to query IMDS rather than
  // evidence that the peer is AWS, it defaults to true off EC2 where no IMDS
  // exists, and a region it does yield is the instance's, not the bucket's. A
  // DNS alias fronting AWS falls here too and needs hive.s3.endpoint.region
  // configured explicitly.
  return isAWSEndpoint(endpoint)
      ? std::optional<std::string>(kS3AwsGlobalRegion)
      : std::nullopt;
}

std::optional<std::string> parseAWSStandardRegionName(
    std::string_view endpoint) {
  if (!isAWSEndpoint(endpoint)) {
    return std::nullopt;
  }
  endpoint = withoutTrailingSlash(endpoint);
  // Remove the kAmazonHostSuffix.
  std::string_view endpointPrefix =
      endpoint.substr(0, endpoint.size() - kAmazonHostSuffix.size());
  const re2::RE2 pattern("^(?:.+\\.)?s3[-.]([a-z0-9-]+)$");
  std::string region;
  if (re2::RE2::FullMatch(endpointPrefix, pattern, &region)) {
    // endpointPrefix is 'bucket.s3-[region]' or 'bucket.s3.[region]'
    return region;
  }

  auto index = endpointPrefix.rfind('.');
  if (index != std::string::npos) {
    // endpointPrefix was 'service.[region]'.
    return std::string(endpointPrefix.substr(index + 1));
  }

  // Use default region set by the SDK.
  return std::nullopt;
}

} // namespace facebook::velox::filesystems
