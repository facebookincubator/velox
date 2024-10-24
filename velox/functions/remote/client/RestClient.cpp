#include "velox/functions/remote/client/RestClient.h"
#include "velox/exec/ExchangeQueue.h"

namespace facebook::velox::functions {

//
// RestClient Implementation
//

RestClient::RestClient(
    const std::string& url,
    const std::unordered_map<std::string, std::string>& headers)
    : url_(proxygen::URL(url)), headers_(headers) {
  httpClient_ = std::make_shared<HttpClient>(url_);
}

std::pair<int, std::unique_ptr<exec::SerializedPage>>
RestClient::invoke_function(exec::SerializedPage& requestPage) {
  httpClient_->setHeaders(headers_);
  httpClient_->send(requestPage);

  // Retrieve the response page as a unique_ptr
  auto responsePage = httpClient_->getResponsePage();

  int statusCode = httpClient_->getResponseCode();

  return {statusCode, std::move(responsePage)};
}

//
// HttpClient Implementation
//

HttpClient::HttpClient(const proxygen::URL& url)
    : url_(url), responseCode_(0) {}

void HttpClient::setHeaders(
    const std::unordered_map<std::string, std::string>& headers) {
  headers_ = headers;
}

void HttpClient::send(const exec::SerializedPage& serializedPage) {
  // Get the IOBuf from SerializedPage
  requestBodyIOBuf_ = serializedPage.getIOBuf();

  responseBodyIOBuf_.reset();
  responseCode_ = 0;

  // Reset connector and session for resending the request
  connector_.reset();
  session_.reset();

  // Create a new connector for the request
  connector_ = std::make_unique<proxygen::HTTPConnector>(
      this, proxygen::WheelTimerInstance(std::chrono::milliseconds(1000)));

  // Initiate connection
  connector_->connect(
      &evb_,
      folly::SocketAddress(url_.getHost(), url_.getPort(), true),
      std::chrono::milliseconds(10000));

  // Run the event loop until we explicitly terminate it
  evb_.loopForever();
}

std::unique_ptr<exec::SerializedPage> HttpClient::getResponsePage() {
  if (responseBodyIOBuf_) {
    // Construct SerializedPage using the response IOBuf
    return std::make_unique<exec::SerializedPage>(
        std::move(responseBodyIOBuf_));
  } else {
    // Return nullptr or handle error
    return nullptr;
  }
}

int HttpClient::getResponseCode() const {
  return responseCode_;
}

// HTTPConnector::Callback methods
void HttpClient::connectSuccess(
    proxygen::HTTPUpstreamSession* session) noexcept {
  session_ = std::shared_ptr<proxygen::HTTPUpstreamSession>(
      session, [](proxygen::HTTPUpstreamSession* /*s*/) {
        // No-op deleter, session is managed by Proxygen
      });
  sendRequest();
}

void HttpClient::connectError(
    const folly::AsyncSocketException& ex) noexcept {
  LOG(ERROR) << "Failed to connect: " << ex.what();
  evb_.terminateLoopSoon();
}

// HTTPTransactionHandler methods
void HttpClient::setTransaction(
    proxygen::HTTPTransaction* txn) noexcept {
  txn_ = txn;
}

void HttpClient::detachTransaction() noexcept {
  txn_ = nullptr;
  session_.reset();
  evb_.terminateLoopSoon();
}

void HttpClient::onHeadersComplete(
    std::unique_ptr<proxygen::HTTPMessage> msg) noexcept {
  responseCode_ = msg->getStatusCode();
}

void HttpClient::onBody(
    std::unique_ptr<folly::IOBuf> chain) noexcept {
  if (chain) {
    if (responseBodyIOBuf_) {
      responseBodyIOBuf_->prependChain(std::move(chain));
    } else {
      responseBodyIOBuf_ = std::move(chain);
    }
  }
}

void HttpClient::onEOM() noexcept {
  evb_.terminateLoopSoon();
}

void HttpClient::onError(
    const proxygen::HTTPException& error) noexcept {
  LOG(ERROR) << "HTTP Error: " << error.what();
  evb_.terminateLoopSoon();
}

void HttpClient::sendRequest() {
  auto txn = session_->newTransaction(this);
  if (!txn) {
    LOG(ERROR) << "Failed to create new transaction";
    evb_.terminateLoopSoon();
    return;
  }

  proxygen::HTTPMessage req;
  req.setMethod(proxygen::HTTPMethod::POST);
  req.setURL(url_.makeRelativeURL());

  req.getHeaders().add("Host", url_.getHostAndPort());
  for (const auto& header : headers_) {
    req.getHeaders().add(header.first, header.second);
  }

  txn->sendHeaders(req);
  txn->sendBody(std::move(requestBodyIOBuf_));
  txn->sendEOM();
}

} // namespace facebook::velox::functions
