@echo off

taskkill /IM scrcpy.exe /F
taskkill /IM adb.exe /F

adb kill-server
adb start-server

scrcpy --max-size 960
