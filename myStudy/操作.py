import json

from 运行辅助 import MyMNTDevice
_DEVICE_ID = 'emulator-5554'
设备 = MyMNTDevice(_DEVICE_ID)
with open('json/名称_操作.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
# for name, value in data.items():
#     print(name, value)
#     设备.发送(value)
#     设备.tap([(62, 441)])
设备.发送('v 1\n^ 10 2159 1079 2\n$ 16685\n')
设备.stop()