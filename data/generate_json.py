# -*- coding: utf-8 -*-
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

import sys
import json
import copy


class JsonGen:
    def __init__(self, data):
        self.__dict__ = copy.deepcopy(data)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii = False)

    def repeat(self, num):
        json_arr = []
        tmp_dict = copy.deepcopy(self.__dict__)
        for i in range(num):
            tmp_dict["id"] = tmp_dict["id"] + 1
            json_arr.append(tmp_dict)
        self.__dict__.clear()
        self.__dict__["statuses"] = json_arr

    def dict(self):
        return self.__dict__


if __name__ == "__main__":
    data = {
        "metadata": {
            "result_type": "recent", 
            "iso_language_code": "ja"
        },
        "created_at": "Sun Aug 31 00:29:15 +0000 2014",
        "id": 10000000000,
        "text": "@aym0566x \n\n名前:第一:なんか怖っ！\n今の印象:噛み合い\nところ:ぶすでキモいとこ😋✨✨\n思い出:んーーー、ありすぎ😊❤️\n",
        "in_reply_to_user_id_str": "866260188",
        "in_reply_to_screen_name": "aym0566x",
        "user": {
            "id": 1186275104,
            "id_str": "1186275104",
            "name": "AYUMI",
            "screen_name": "ayuu0123",
            "location": "",
            "entities": {
              "description": {
                "urls": []
              }
            }
        },
        "followers_count": 262,
        "friends_count": 252,
        "listed_count": 0,
        "created_at": "Sat Feb 16 13:40:25 +0000 2013"
    }

    repeat_num = [1,10,100,1000,10000]
    for i in repeat_num:
        OriJson = JsonGen(data)
        OriJson.repeat(i)
        OriJson.save("%s%s%s" % ("twitter_", i, "K.json"))  # save repeat file
