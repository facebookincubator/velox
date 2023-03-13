# -*- coding: utf-8 -*-
import sys
import json
import pandas as pd


class JsonGen():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)  
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4) 

    def repeat(self, num):
        json_arr = []
        for i in range(num):
            self.__dict__["id"] = self.__dict__["id"] + 1
            json_str = json.dumps(self.__dict__, indent=4)
            json_arr.append(json_str)
        self.__dict__.clear()
        self.__dict__["statuses"] = json_arr

    def dict(self):
        return self.__dict__


if __name__ == '__main__':
    repeat_num = int(sys.argv[1])

    data = {
      "metadata": {
        "result_type": "recent",
        "iso_language_code": "ja"
      },
      "created_at": "Sun Aug 31 00:29:15 +0000 2014",
      "id": 100000000,
      "text": "@aym0566x \n\n名前:第一:なんか怖っ！\n今の印象:噛み合い\nところ:ぶすでキモいとこ😋✨✨\n思い出:んーーー、ありすぎ😊❤️\nLINE交換できる？:あぁ……ごめん✋\nトプ画をみて:照れますがな😘✨\n一言:お前は一生もんのダチ💖",
      "in_reply_to_user_id_str": "866260188",
      "in_reply_to_screen_name": "aym0566x"
    }
    json_str = json.dumps(data, indent=4)

    with open('twitter_1K.json', 'w') as f:  # 创建一个params.json文件
        f.write(json_str)  # 将json_str写到文件中
    OriJson = JsonGen('twitter_1K.json')
    OriJson.repeat(repeat_num)
    OriJson.save("%s%s%s"%('twitter_',repeat_num,"K.json"))  # 将修改后的数据保存
