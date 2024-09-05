import time
from time import sleep

import json
from 运行辅助 import MyMNTDevice

yy, xx = 500, 220
# xx 和模拟器相比是x2了的
data = {
    "上移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy + 60} {xx} 100\nc\n",
    "右移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy} {xx + 60} 100\nc\n",
    "下移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy - 60} {xx} 100\nc\n",
    "左移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy} {xx - 60} 100\nc\n",
    "左上移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy + 240} {xx - 60} 100\nc\n",
    "左下移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy - 240} {xx - 60} 100\nc\n",
    "右上移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy + 240} {xx + 60} 100\nc\n",
    "右下移": f"d 1 {yy} {xx} 300\nc\nm 1 {yy - 240} {xx + 60} 100\nc\n",
    "移动停": "u 1\nc\n",
}

device = MyMNTDevice('emulator-5554')

print("max x:", device.connection.max_x)
print("max y:", device.connection.max_y)
i = 0
for _ in range(1):
    for key, value in data.items():
        print(key)
        # device.发送(value)
        # time.sleep(1)
        # device.发送(data['移动停'])
        device.发送(f"d 0 {2159 - 1876} {1921} 100\nc\nu 0\nc\n")
        sleep(1)
        device.发送(f"d 1 {2159 - 1876} {921} 100\nc\nu 1\nc\n")
        sleep(1)
print(json.dumps(data, ensure_ascii=False, indent=2))
device.stop()
