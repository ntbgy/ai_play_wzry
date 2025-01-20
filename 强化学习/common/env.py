import json
import os

project_root_path = r"C:\Users\ntbgy\PycharmProjects\ai-play-wzry\强化学习"
project_log_path = r"%temp%\强化学习\log"
device_id = 'emulator-5554'
scrcpy_windows_name = "scrcpy投屏"
# 模拟器比真机要慢亿点点
sleep_time = 1
load_weights = 'E:/ai-play-wzry/weights'
模型名称 = 'model_weights_5v5_1v1.pth'
保存模型名称 = 'model_weights_5v5_1v1.pth'
training_data_save_directory = 'E:/ai-play-wzry/训练数据样本/未用'
判断状态模型地址 = r"E:/ai-play-wzry/weights/model_weights_judgment_state.pth"
状态词典 = {
    "击杀小兵或野怪或推掉塔": 2,
    "击杀敌方英雄": 5,
    "被击塔攻击": -1,
    "被击杀": -2,
    "无状况": 0.01,
    "死亡": -0.05,
    "其它": 0.001,
    "普通": 0.01,
    "胜利": 50,
    "失败": -10,
}
状态词典B = {
    "击杀小兵或野怪或推掉塔": 0,
    "击杀敌方英雄": 1,
    "被击塔攻击": 2,
    "被击杀": 3,
    "死亡": 4,
    "普通": 5,
    "胜利": 6,
    "失败": 7
}
操作查询路径 = os.path.join(project_root_path, "json/名称_操作.json")
with open(操作查询路径, encoding='utf8') as f:
    操作查询词典 = json.load(f)
操作词典 = {"图片号": "0", "移动操作": "无移动", "动作操作": "无动作"}
词数词典路径 = os.path.join(project_root_path, "json/词_数表.json")
数_词表路径 = os.path.join(project_root_path, "json/数_词表.json")
