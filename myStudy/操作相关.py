import json
import time

from common.env import device_id
from 运行辅助 import MyMNTDevice

device = MyMNTDevice(device_id)


def 操作测试1():
    with open('../json/名称_操作.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("max x:", device.connection.max_x)
    print("max y:", device.connection.max_y)
    for name, value in data.items():
        if isinstance(value, list):
            print(name, value)
            # 转换算法离大普
            x, y = value
            x, y = 转换(x, y)
            value = f"d 0 {x} {y} 100\nc\nu 0\nc\n"
            print(value)
            data[name] = value
            time.sleep(1)
            device.发送(value)
            time.sleep(2)
    device.stop()


def 转换(x, y):
    x, y = y, x
    x = 3199 - int(x * (3199 / 1439))
    y = int(y * (1439 / 3199))
    return x, y


def 操作转写():
    with open('../json/名称_操作.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for name, value in data.items():
        if isinstance(value, list):
            print(name, value)
            x, y = value
            x, y = 转换(x, y)
            value = f"d 0 {x} {y} 100\nc\nu 0\nc\n"
            print(value)
            data[name] = value
    with open('../json/名称_操作.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def 操作测试2():
    with open('../json/名称_操作.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("max x:", device.connection.max_x)
    print("max y:", device.connection.max_y)
    for _ in range(1):
        for key, value in data.items():
            print(key)
            time.sleep(2)
            device.发送(value)
            time.sleep(3)
    device.stop()


def 移动测试():
    """
    max x: 3199
    max y: 1439
    """
    x, y = 624, 1439 - 1142

    data = {
        "上移": f"d 1 {x} {y} 500\nc\nm 1 {x + 100} {y} 100\nc\n",
        "右移": f"d 1 {x} {y} 500\nc\nm 1 {x} {y + 100} 100\nc\n",
        "下移": f"d 1 {x} {y} 500\nc\nm 1 {x - 100} {y} 100\nc\n",
        "左移": f"d 1 {x} {y} 500\nc\nm 1 {x} {y - 100} 100\nc\n",
        "左上移": f"d 1 {x} {y} 500\nc\nm 1 {x + 400} {y - 100} 100\nc\n",
        "左下移": f"d 1 {x} {y} 500\nc\nm 1 {x - 400} {y - 100} 100\nc\n",
        "右上移": f"d 1 {x} {y} 500\nc\nm 1 {x + 400} {y + 100} 100\nc\n",
        "右下移": f"d 1 {x} {y} 500\nc\nm 1 {x - 400} {y + 100} 100\nc\n",
        "移动停": "u 1\nc\n",
    }

    print("max x:", device.connection.max_x)
    print("max y:", device.connection.max_y)
    for _ in range(1):
        for key, value in data.items():
            print(key)
            device.发送(value)
            time.sleep(3)
            device.发送(data['移动停'])
    print(json.dumps(data, ensure_ascii=False, indent=2))
    device.stop()


if __name__ == '__main__':
    # 操作转写()
    # 移动测试()
    操作测试2()
