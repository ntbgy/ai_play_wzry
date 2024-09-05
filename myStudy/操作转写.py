import json
# from 移动测试 import value
#
# from 运行辅助 import MyMNTDevice
# _DEVICE_ID = 'emulator-5554'
# 设备 = MyMNTDevice(_DEVICE_ID)
with open('json/名称_操作.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for name, value in data.items():
    if isinstance(value, list):
        print(name, value)
        x,y=value
        value = f"d 0 {2159 - x} {y} 100\nc\nu 0\nc\n"
        print(value)
        data[name] = value
with open('json/名称_操作_2.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
# 设备.stop()