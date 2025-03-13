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

#include "velox/functions/remote/utils/restserver/RemoteFunctionRestService.h"

#include <boost/beast/version.hpp>
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/fbhive/HiveTypeParser.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::functions {

RestSession::RestSession(
    boost::asio::ip::tcp::socket socket,
    std::string functionPrefix)
    : socket_(std::move(socket)) {
  if (!pool_) {
    pool_ = memory::memoryManager()->addLeafPool();
  }
}

void RestSession::run() {
  doRead();
}

void RestSession::doRead() {
  auto self = shared_from_this();
  boost::beast::http::async_read(
      socket_,
      buffer_,
      req_,
      [self](boost::beast::error_code ec, std::size_t bytes_transferred) {
        self->onRead(ec, bytes_transferred);
      });
}

void RestSession::onRead(
    boost::beast::error_code ec,
    std::size_t bytes_transferred) {
  boost::ignore_unused(bytes_transferred);

  if (ec == boost::beast::http::error::end_of_stream) {
    return doClose();
  }

  if (ec) {
    LOG(ERROR) << "Read error: " << ec.message();
    return;
  }

  handleRequest(std::move(req_));
}

void RestSession::handleRequest(
    boost::beast::http::request<boost::beast::http::string_body> req) {
  // Prepare the response (common headers, etc.)
  res_.version(req.version());
  res_.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);

  // Only POST is allowed.
  if (req.method() != boost::beast::http::verb::post) {
    res_.result(boost::beast::http::status::method_not_allowed);
    res_.set(boost::beast::http::field::content_type, "text/plain");
    res_.body() = "Only POST method is allowed";
    res_.prepare_payload();

    auto self = shared_from_this();
    boost::beast::http::async_write(
        socket_,
        res_,
        [self](boost::beast::error_code ec, std::size_t bytes_transferred) {
          self->onWrite(true, ec, bytes_transferred);
        });
    return;
  }

  std::string path = req.target();

  // Expected path format:
  // /{functionName}
  std::vector<std::string> pathComponents;
  folly::split('/', path, pathComponents);

  std::string functionName;
  if (pathComponents.size() <= 2) {
    functionName = pathComponents[1];
  } else {
    res_.result(boost::beast::http::status::bad_request);
    res_.set(boost::beast::http::field::content_type, "text/plain");
    res_.body() = "Invalid request path";
    res_.prepare_payload();

    auto self = shared_from_this();
    boost::beast::http::async_write(
        socket_,
        res_,
        [self](boost::beast::error_code ec, std::size_t bytes_transferred) {
          self->onWrite(true, ec, bytes_transferred);
        });
    return;
  }

  try {
    if (functionName != "remote_abs") {
      VELOX_USER_FAIL(
          "Function '{}' is not available on the server.", functionName);
    }

    // We assume a single argument of type INTEGER and a single return type of
    // INTEGER. If you have these from some request headers, you can parse them;
    // otherwise, if you know them upfront, you can hard-code them:
    std::vector<std::string> argTypeNames = {"integer"};
    std::string returnTypeName = "integer";

    serializer::presto::PrestoVectorSerde serde;
    auto inputBuffer = folly::IOBuf::copyBuffer(req.body());

    // Parse the input argument type and output type (both should be INTEGER
    // here).
    auto argType = type::fbhive::HiveTypeParser().parse(argTypeNames[0]);
    auto outType = type::fbhive::HiveTypeParser().parse(returnTypeName);

    VELOX_CHECK_EQ(
        argType->kind(),
        outType->kind(),
        "For this simple 'abs' function, argument type and return type should match.");

    // Build the row type with a single child of type argType.
    auto rowType = ROW({argType});
    auto inputVector = IOBufToRowVector(*inputBuffer, rowType, *pool_, &serde);

    VELOX_CHECK_EQ(
        inputVector->childrenSize(),
        1,
        "Expected exactly 1 column for function '{}'.",
        functionName);

    // Create the result vector.
    const auto numRows = inputVector->size();
    auto resultVector = BaseVector::create(argType, numRows, pool_.get());

    // Compute abs for INTEGER.
    auto inputFlat = inputVector->childAt(0)->asFlatVector<int32_t>();
    auto outFlat = resultVector->asFlatVector<int32_t>();
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (inputFlat->isNullAt(i)) {
        outFlat->setNull(i, true);
      } else {
        int32_t val = inputFlat->valueAt(i);
        outFlat->set(i, std::abs(val));
      }
    }

    // Wrap in a RowVector for serialization.
    auto outputRowVector = std::make_shared<RowVector>(
        pool_.get(),
        rowType,
        BufferPtr(),
        numRows,
        std::vector<VectorPtr>{resultVector});

    // Serialize the result back to an IOBuf.
    auto payload = rowVectorToIOBuf(
        outputRowVector, outputRowVector->size(), *pool_, &serde);

    // Construct a successful response.
    res_.result(boost::beast::http::status::ok);
    res_.set(
        boost::beast::http::field::content_type, "application/octet-stream");
    res_.body() = payload.moveToFbString().toStdString();
    res_.prepare_payload();

    auto self = shared_from_this();
    boost::beast::http::async_write(
        socket_,
        res_,
        [self](boost::beast::error_code ec, std::size_t bytes_transferred) {
          self->onWrite(false, ec, bytes_transferred);
        });

  } catch (const std::exception& ex) {
    // Handle any errors that occurred in the above logic.
    LOG(ERROR) << ex.what();
    res_.result(boost::beast::http::status::internal_server_error);
    res_.set(boost::beast::http::field::content_type, "text/plain");
    res_.body() = ex.what();
    res_.prepare_payload();

    auto self = shared_from_this();
    boost::beast::http::async_write(
        socket_,
        res_,
        [self](boost::beast::error_code ec, std::size_t bytes_transferred) {
          self->onWrite(true, ec, bytes_transferred);
        });
  }
}

void RestSession::onWrite(
    bool close,
    boost::beast::error_code ec,
    std::size_t bytes_transferred) {
  boost::ignore_unused(bytes_transferred);

  if (ec) {
    LOG(ERROR) << "Write error: " << ec.message();
    return;
  }

  if (close) {
    return doClose();
  }

  req_ = {};

  doRead();
}

void RestSession::doClose() {
  boost::beast::error_code ec;
  socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);
}

RestListener::RestListener(
    boost::asio::io_context& ioc,
    boost::asio::ip::tcp::endpoint endpoint,
    std::string functionPrefix)
    : ioc_(ioc), acceptor_(ioc), functionPrefix_(std::move(functionPrefix)) {
  boost::beast::error_code ec;

  acceptor_.open(endpoint.protocol(), ec);
  if (ec) {
    LOG(ERROR) << "Open error: " << ec.message();
    return;
  }

  acceptor_.set_option(boost::asio::socket_base::reuse_address(true), ec);
  if (ec) {
    LOG(ERROR) << "Set_option error: " << ec.message();
    return;
  }

  acceptor_.bind(endpoint, ec);
  if (ec) {
    LOG(ERROR) << "Bind error: " << ec.message();
    return;
  }

  acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
  if (ec) {
    LOG(ERROR) << "Listen error: " << ec.message();
    return;
  }
}

void RestListener::run() {
  doAccept();
}

void RestListener::doAccept() {
  acceptor_.async_accept(
      [self = shared_from_this()](
          boost::beast::error_code ec, boost::asio::ip::tcp::socket socket) {
        self->onAccept(ec, std::move(socket));
      });
}

void RestListener::onAccept(
    boost::beast::error_code ec,
    boost::asio::ip::tcp::socket socket) {
  if (ec) {
    LOG(ERROR) << "Accept error: " << ec.message();
  } else {
    std::make_shared<RestSession>(std::move(socket), functionPrefix_)->run();
  }
  doAccept();
}

} // namespace facebook::velox::functions
