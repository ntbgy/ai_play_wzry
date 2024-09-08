@echo off

taskkill /IM scrcpy.exe /F
taskkill /IM adb.exe /F

adb kill-server
adb start-server

scrcpy -s emulator-5554 --max-size 960
scrcpy -s emulator-5556 --max-size 960
