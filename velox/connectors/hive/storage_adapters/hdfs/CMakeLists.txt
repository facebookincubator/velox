# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# for generated headers

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
add_library(velox_hdfs HdfsFileSystem.cpp HdfsReadFile.cpp)
target_link_libraries(velox_hdfs velox_hdfs3 consul)
add_subdirectory(service_discover)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libhdfs3/CMake"
                      ${CMAKE_MODULE_PATH})

set(HAVE_BOOST_CHRONO 0)
set(HAVE_BOOST_ATOMIC 0)

set(HAVE_STD_CHRONO 1)
set(HAVE_STD_ATOMIC 1)

# project and source dir
set(HDFS3_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libhdfs3/src)
add_compile_definitions(HAVE_FEATURES_H=0)
add_definitions(-DBOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED)
if(APPLE)
  add_definitions(-DOS_MACOSX)
endif()
add_definitions(-DHAVE_DLADDR)
add_definitions(-DSTRERROR_R_RETURN_INT)

# generated code
set(PROTO_TARGETS
    ${HDFS3_SOURCE_DIR}/proto/encryption.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/ClientDatanodeProtocol.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/hdfs.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/Security.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/ProtobufRpcEngine.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/ClientNamenodeProtocol.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/IpcConnectionContext.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/RpcHeader.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/datatransfer.pb.cc
    ${HDFS3_SOURCE_DIR}/proto/encryption.pb.h
    ${HDFS3_SOURCE_DIR}/proto/ClientDatanodeProtocol.pb.h
    ${HDFS3_SOURCE_DIR}/proto/hdfs.pb.h
    ${HDFS3_SOURCE_DIR}/proto/Security.pb.h
    ${HDFS3_SOURCE_DIR}/proto/ProtobufRpcEngine.pb.h
    ${HDFS3_SOURCE_DIR}/proto/ClientNamenodeProtocol.pb.h
    ${HDFS3_SOURCE_DIR}/proto/IpcConnectionContext.pb.h
    ${HDFS3_SOURCE_DIR}/proto/RpcHeader.pb.h
    ${HDFS3_SOURCE_DIR}/proto/datatransfer.pb.h)

add_custom_command(
  OUTPUT ${PROTO_TARGETS}
  COMMAND protoc "${HDFS3_SOURCE_DIR}/proto/*.proto" -I
          "${HDFS3_SOURCE_DIR}/proto" --cpp_out="${HDFS3_SOURCE_DIR}/proto")

configure_file(${HDFS3_SOURCE_DIR}/platform.h.in ${HDFS3_SOURCE_DIR}/platform.h)

set(SRCS
    ${HDFS3_SOURCE_DIR}/network/TcpSocket.cpp
    ${HDFS3_SOURCE_DIR}/network/DomainSocket.cpp
    ${HDFS3_SOURCE_DIR}/network/BufferedSocketReader.cpp
    ${HDFS3_SOURCE_DIR}/client/ReadShortCircuitInfo.cpp
    ${HDFS3_SOURCE_DIR}/client/Pipeline.cpp
    ${HDFS3_SOURCE_DIR}/client/Hdfs.cpp
    ${HDFS3_SOURCE_DIR}/client/Packet.cpp
    ${HDFS3_SOURCE_DIR}/client/OutputStreamImpl.cpp
    ${HDFS3_SOURCE_DIR}/client/KerberosName.cpp
    ${HDFS3_SOURCE_DIR}/client/PacketHeader.cpp
    ${HDFS3_SOURCE_DIR}/client/LocalBlockReader.cpp
    ${HDFS3_SOURCE_DIR}/client/UserInfo.cpp
    ${HDFS3_SOURCE_DIR}/client/RemoteBlockReader.cpp
    ${HDFS3_SOURCE_DIR}/client/Permission.cpp
    ${HDFS3_SOURCE_DIR}/client/FileSystemImpl.cpp
    ${HDFS3_SOURCE_DIR}/client/DirectoryIterator.cpp
    ${HDFS3_SOURCE_DIR}/client/FileSystemKey.cpp
    ${HDFS3_SOURCE_DIR}/client/DataTransferProtocolSender.cpp
    ${HDFS3_SOURCE_DIR}/client/LeaseRenewer.cpp
    ${HDFS3_SOURCE_DIR}/client/PeerCache.cpp
    ${HDFS3_SOURCE_DIR}/client/InputStream.cpp
    ${HDFS3_SOURCE_DIR}/client/FileSystem.cpp
    ${HDFS3_SOURCE_DIR}/client/InputStreamImpl.cpp
    ${HDFS3_SOURCE_DIR}/client/Token.cpp
    ${HDFS3_SOURCE_DIR}/client/PacketPool.cpp
    ${HDFS3_SOURCE_DIR}/client/OutputStream.cpp
    ${HDFS3_SOURCE_DIR}/client/KmsClientProvider.cpp
    ${HDFS3_SOURCE_DIR}/client/DataReader.cpp
    ${HDFS3_SOURCE_DIR}/client/CryptoCodec.cpp
    ${HDFS3_SOURCE_DIR}/client/EncryptionZoneIterator.cpp
    ${HDFS3_SOURCE_DIR}/client/HttpClient.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcChannelKey.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcProtocolInfo.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcClient.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcRemoteCall.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcChannel.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcAuth.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcContentWrapper.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcConfig.cpp
    ${HDFS3_SOURCE_DIR}/rpc/RpcServerInfo.cpp
    ${HDFS3_SOURCE_DIR}/rpc/SaslClient.cpp
    ${HDFS3_SOURCE_DIR}/rpc/DigestMD5.cpp
    ${HDFS3_SOURCE_DIR}/server/Datanode.cpp
    ${HDFS3_SOURCE_DIR}/server/LocatedBlocks.cpp
    ${HDFS3_SOURCE_DIR}/server/NamenodeProxy.cpp
    ${HDFS3_SOURCE_DIR}/server/NamenodeImpl.cpp
    ${HDFS3_SOURCE_DIR}/server/NamenodeInfo.cpp
    ${HDFS3_SOURCE_DIR}/common/WritableUtils.cpp
    ${HDFS3_SOURCE_DIR}/common/ExceptionInternal.cpp
    ${HDFS3_SOURCE_DIR}/common/SessionConfig.cpp
    ${HDFS3_SOURCE_DIR}/common/StackPrinter.cpp
    ${HDFS3_SOURCE_DIR}/common/Exception.cpp
    ${HDFS3_SOURCE_DIR}/common/Logger.cpp
    ${HDFS3_SOURCE_DIR}/common/CFileWrapper.cpp
    ${HDFS3_SOURCE_DIR}/common/XmlConfig.cpp
    ${HDFS3_SOURCE_DIR}/common/WriteBuffer.cpp
    ${HDFS3_SOURCE_DIR}/common/HWCrc32c.cpp
    ${HDFS3_SOURCE_DIR}/common/MappedFileWrapper.cpp
    ${HDFS3_SOURCE_DIR}/common/Hash.cpp
    ${HDFS3_SOURCE_DIR}/common/SWCrc32c.cpp
    ${HDFS3_SOURCE_DIR}/common/Thread.cpp
    ${HDFS3_SOURCE_DIR}/client/NodeResolver.cpp
    ${HDFS3_SOURCE_DIR}/network/TcpSocket.h
    ${HDFS3_SOURCE_DIR}/network/BufferedSocketReader.h
    ${HDFS3_SOURCE_DIR}/network/Socket.h
    ${HDFS3_SOURCE_DIR}/network/DomainSocket.h
    ${HDFS3_SOURCE_DIR}/network/Syscall.h
    ${HDFS3_SOURCE_DIR}/client/InputStreamImpl.h
    ${HDFS3_SOURCE_DIR}/client/FileSystem.h
    ${HDFS3_SOURCE_DIR}/client/ReadShortCircuitInfo.h
    ${HDFS3_SOURCE_DIR}/client/InputStreamInter.h
    ${HDFS3_SOURCE_DIR}/client/FileSystemImpl.h
    ${HDFS3_SOURCE_DIR}/client/PacketPool.h
    ${HDFS3_SOURCE_DIR}/client/Pipeline.h
    ${HDFS3_SOURCE_DIR}/client/OutputStreamInter.h
    ${HDFS3_SOURCE_DIR}/client/RemoteBlockReader.h
    ${HDFS3_SOURCE_DIR}/client/Token.h
    ${HDFS3_SOURCE_DIR}/client/KerberosName.h
    ${HDFS3_SOURCE_DIR}/client/DirectoryIterator.h
    ${HDFS3_SOURCE_DIR}/client/hdfs.h
    ${HDFS3_SOURCE_DIR}/client/FileSystemStats.h
    ${HDFS3_SOURCE_DIR}/client/FileSystemKey.h
    ${HDFS3_SOURCE_DIR}/client/DataTransferProtocolSender.h
    ${HDFS3_SOURCE_DIR}/client/Packet.h
    ${HDFS3_SOURCE_DIR}/client/PacketHeader.h
    ${HDFS3_SOURCE_DIR}/client/FileSystemInter.h
    ${HDFS3_SOURCE_DIR}/client/LocalBlockReader.h
    ${HDFS3_SOURCE_DIR}/client/TokenInternal.h
    ${HDFS3_SOURCE_DIR}/client/InputStream.h
    ${HDFS3_SOURCE_DIR}/client/PipelineAck.h
    ${HDFS3_SOURCE_DIR}/client/BlockReader.h
    ${HDFS3_SOURCE_DIR}/client/Permission.h
    ${HDFS3_SOURCE_DIR}/client/OutputStreamImpl.h
    ${HDFS3_SOURCE_DIR}/client/LeaseRenewer.h
    ${HDFS3_SOURCE_DIR}/client/UserInfo.h
    ${HDFS3_SOURCE_DIR}/client/PeerCache.h
    ${HDFS3_SOURCE_DIR}/client/OutputStream.h
    ${HDFS3_SOURCE_DIR}/client/FileStatus.h
    ${HDFS3_SOURCE_DIR}/client/DataTransferProtocol.h
    ${HDFS3_SOURCE_DIR}/client/BlockLocation.h
    ${HDFS3_SOURCE_DIR}/client/KmsClientProvider.h
    ${HDFS3_SOURCE_DIR}/client/DataReader.h
    ${HDFS3_SOURCE_DIR}/client/CryptoCodec.h
    ${HDFS3_SOURCE_DIR}/client/EncryptionZoneIterator.h
    ${HDFS3_SOURCE_DIR}/client/HttpClient.h
    ${HDFS3_SOURCE_DIR}/client/NodeResolver.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcConfig.h
    ${HDFS3_SOURCE_DIR}/rpc/SaslClient.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcAuth.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcClient.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcCall.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcContentWrapper.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcProtocolInfo.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcRemoteCall.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcServerInfo.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcChannel.h
    ${HDFS3_SOURCE_DIR}/rpc/RpcChannelKey.h
    ${HDFS3_SOURCE_DIR}/rpc/DigestMD5.h
    ${HDFS3_SOURCE_DIR}/server/BlockLocalPathInfo.h
    ${HDFS3_SOURCE_DIR}/server/LocatedBlocks.h
    ${HDFS3_SOURCE_DIR}/server/DatanodeInfo.h
    ${HDFS3_SOURCE_DIR}/server/RpcHelper.h
    ${HDFS3_SOURCE_DIR}/server/ExtendedBlock.h
    ${HDFS3_SOURCE_DIR}/server/NamenodeInfo.h
    ${HDFS3_SOURCE_DIR}/server/NamenodeImpl.h
    ${HDFS3_SOURCE_DIR}/server/LocatedBlock.h
    ${HDFS3_SOURCE_DIR}/server/NamenodeProxy.h
    ${HDFS3_SOURCE_DIR}/server/Datanode.h
    ${HDFS3_SOURCE_DIR}/server/Namenode.h
    ${HDFS3_SOURCE_DIR}/server/EncryptionKey.h
    ${HDFS3_SOURCE_DIR}/common/XmlConfig.h
    ${HDFS3_SOURCE_DIR}/common/Logger.h
    ${HDFS3_SOURCE_DIR}/common/WriteBuffer.h
    ${HDFS3_SOURCE_DIR}/common/HWCrc32c.h
    ${HDFS3_SOURCE_DIR}/common/Checksum.h
    ${HDFS3_SOURCE_DIR}/common/SessionConfig.h
    ${HDFS3_SOURCE_DIR}/common/Unordered.h
    ${HDFS3_SOURCE_DIR}/common/BigEndian.h
    ${HDFS3_SOURCE_DIR}/common/Thread.h
    ${HDFS3_SOURCE_DIR}/common/StackPrinter.h
    ${HDFS3_SOURCE_DIR}/common/Exception.h
    ${HDFS3_SOURCE_DIR}/common/WritableUtils.h
    ${HDFS3_SOURCE_DIR}/common/StringUtil.h
    ${HDFS3_SOURCE_DIR}/common/LruMap.h
    ${HDFS3_SOURCE_DIR}/common/Function.h
    ${HDFS3_SOURCE_DIR}/common/DateTime.h
    ${HDFS3_SOURCE_DIR}/common/Hash.h
    ${HDFS3_SOURCE_DIR}/common/SWCrc32c.h
    ${HDFS3_SOURCE_DIR}/common/ExceptionInternal.h
    ${HDFS3_SOURCE_DIR}/common/Memory.h
    ${HDFS3_SOURCE_DIR}/common/FileWrapper.h)

# target
add_library(velox_hdfs3 STATIC ${SRCS} ${PROTO_TARGETS})

# include and linkvelox_hdfs3
target_include_directories(velox_hdfs3 PRIVATE ${HDFS3_SOURCE_DIR})
target_include_directories(velox_hdfs3 PRIVATE ${HDFS3_SOURCE_DIR}/proto)
target_include_directories(velox_hdfs3 PRIVATE ${HDFS3_SOURCE_DIR}/client)
target_include_directories(velox_hdfs3 PRIVATE ${HDFS3_SOURCE_DIR}/server)
target_include_directories(velox_hdfs3 PRIVATE ${HDFS3_SOURCE_DIR}/common)

# todo: setup script should ensure following libraries are installed
find_package(KERBEROS REQUIRED)
target_include_directories(velox_hdfs3 PRIVATE ${KERBEROS_INCLUDE_DIRS})
target_link_libraries(velox_hdfs3 ${KERBEROS_LIBRARIES})

find_package(CURL REQUIRED)
target_include_directories(velox_hdfs3 PRIVATE ${CURL_INCLUDE_DIRS})
target_link_libraries(velox_hdfs3 ${CURL_LIBRARIES})

if(APPLE)
  find_package(LIBXML2 REQUIRED)
else()
  find_package(LibXml2 REQUIRED)
endif()
target_include_directories(velox_hdfs3 PRIVATE ${LIBXML2_INCLUDE_DIR})
target_link_libraries(velox_hdfs3 ${LIBXML2_LIBRARY})

find_package(Protobuf REQUIRED)
target_include_directories(velox_hdfs3 PRIVATE ${Protobuf_INCLUDE_DIR})
target_link_libraries(velox_hdfs3 ${Protobuf_LIBRARY})

find_package(GSasl REQUIRED)
if(APPLE)
  target_include_directories(velox_hdfs3 PRIVATE "/usr/local/include")
  target_link_libraries(velox_hdfs3 "/usr/local/lib/libgsasl.dylib")
else()
  target_include_directories(velox_hdfs3 PRIVATE ${GSASL_INCLUDE_DIR})
  target_link_libraries(velox_hdfs3 ${GSASL_LIBRARIES})
endif()

add_subdirectory(infsec-cmake)
target_link_libraries(velox_hdfs3 infsec)
target_include_directories(velox_hdfs3 PRIVATE infsec/client)

include(find_ssl.cmake)
target_include_directories(velox_hdfs3 PRIVATE ${OPENSSL_INCLUDE_DIR})
target_link_libraries(velox_hdfs3 ${OPENSSL_LIBRARIES})
