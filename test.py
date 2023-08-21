import json
import numpy as np
# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

json_file_path = "/comp_robot/yangjie/edpose_noc/NoC_95_ochuman.json"

data = read_json_file(json_file_path)
mean = np.mean(data)
print(len(data))
print(mean)