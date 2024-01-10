#!/bin/sh
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

set -e

nohup $PRESTO_HOME/bin/launcher --pid-file=/tmp/pidfile  run >  /tmp/server.log
#wait a few seconds for presto to start
sleep 60
echo 'CREATE SCHEMA hive.tpch;' > /tmp/hive_create.sql
/opt/presto-cli --server 127.0.0.1:8080 --file /tmp/hive_create.sql
