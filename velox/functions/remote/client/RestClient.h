#pragma once

#include <folly/init/Init.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <folly/io/async/SSLContext.h>
#include <folly/json.h>
#include <proxygen/lib/http/HTTPConnector.h>
#include <proxygen/lib/http/HTTPMessage.h>
#include <proxygen/lib/http/session/HTTPUpstreamSession.h>
#include <proxygen/lib/utils/URL.h>
#include "velox/exec/ExchangeQueue.h"

namespace facebook::velox::functions {

class HttpClient : public proxygen::HTTPConnector::Callback,
                   public proxygen::HTTPTransactionHandler {
 public:
  explicit HttpClient(const proxygen::URL& url);

  void setHeaders(const std::unordered_map<std::string, std::string>& headers);

  void send(const exec::SerializedPage& serializedPage);

  // Return a unique_ptr to SerializedPage to avoid copy/move
  std::unique_ptr<exec::SerializedPage> getResponsePage();

  int getResponseCode() const;

 private:
  // HTTPConnector::Callback methods
  void connectSuccess(proxygen::HTTPUpstreamSession* session) noexcept override;
  void connectError(const folly::AsyncSocketException& ex) noexcept override;

  // HTTPTransactionHandler methods
  void setTransaction(proxygen::HTTPTransaction* txn) noexcept override;
  void detachTransaction() noexcept override;
  void onHeadersComplete(
      std::unique_ptr<proxygen::HTTPMessage> msg) noexcept override;
  void onBody(std::unique_ptr<folly::IOBuf> chain) noexcept override;
  void onEOM() noexcept override;
  void onError(const proxygen::HTTPException& error) noexcept override;
  void onUpgrade(proxygen::UpgradeProtocol) noexcept override {}
  void onEgressPaused() noexcept override {}
  void onEgressResumed() noexcept override {}
  void onTrailers(std::unique_ptr<proxygen::HTTPHeaders>) noexcept override {}

  void sendRequest();

  proxygen::URL url_;
  folly::EventBase evb_;
  std::unique_ptr<proxygen::HTTPConnector> connector_;
  std::shared_ptr<proxygen::HTTPUpstreamSession> session_;
  std::unordered_map<std::string, std::string> headers_;
  int responseCode_{0};

  // Store request and response bodies as IOBuf pointers
  std::unique_ptr<folly::IOBuf> requestBodyIOBuf_;
  std::unique_ptr<folly::IOBuf> responseBodyIOBuf_;

  // Transaction pointer
  proxygen::HTTPTransaction* txn_{nullptr};
};

class RestClient {
 public:
  RestClient(
      const std::string& url,
      const std::unordered_map<std::string, std::string>& headers = {});

  std::pair<int, std::unique_ptr<exec::SerializedPage>> invoke_function(
      exec::SerializedPage& requestPage);

 private:
  proxygen::URL url_;
  std::unordered_map<std::string, std::string> headers_;
  std::shared_ptr<HttpClient> httpClient_;
};

} // namespace facebook::velox::functions
