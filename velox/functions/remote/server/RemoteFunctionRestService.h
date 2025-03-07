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

#pragma once

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include "velox/expression/FunctionSignature.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::functions {

/// @brief Manages an individual HTTP session.
/// Handles reading HTTP requests, processing them, and sending responses.
/// This class re-hosts Velox functions and allows testing their functionality.
class RestSession : public std::enable_shared_from_this<RestSession>{
 public:
  RestSession(boost::asio::ip::tcp::socket socket, std::string functionPrefix);

  /// Starts the session by initiating a read operation.
  void run();

 private:
  // Initiates an asynchronous read operation.
  void doRead();

  // Called when a read operation completes.
  void onRead(boost::beast::error_code ec, std::size_t bytes_transferred);

  // Processes the HTTP request and prepares a response.
  void handleRequest(
      boost::beast::http::request<boost::beast::http::string_body> req);

  // Called when a write operation completes.
  void onWrite(
      bool close,
      boost::beast::error_code ec,
      std::size_t bytes_transferred);

  // Closes the socket connection.
  void doClose();

  boost::asio::ip::tcp::socket socket_;
  boost::beast::flat_buffer buffer_;
  boost::beast::http::request<boost::beast::http::string_body> req_;
  boost::beast::http::response<boost::beast::http::string_body> res_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

/// @brief Listens for incoming TCP connections and creates sessions.
/// Sets up a TCP acceptor to listen for client connections,
/// creating a new session for each accepted connection.
class RestListener : public std::enable_shared_from_this<RestListener> {
 public:
  RestListener(
      boost::asio::io_context& ioc,
      boost::asio::ip::tcp::endpoint endpoint,
      std::string functionPrefix);

  /// Starts accepting incoming connections.
  void run();

 private:
  // Initiates an asynchronous accept operation.
  void doAccept();

  // Called when an accept operation completes.
  void onAccept(
      boost::beast::error_code ec,
      boost::asio::ip::tcp::socket socket);

  boost::asio::io_context& ioc_;
  boost::asio::ip::tcp::acceptor acceptor_;
  std::string functionPrefix_;
};

} // namespace facebook::velox::functions
