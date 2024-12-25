@echo off
chcp 65001
setlocal enabledelayedexpansion

REM 设置循环次数
set /a loop_count=50

cd D:\yolov10\双模态切换辅助跟随_4
D:

REM 循环开始
for /l %%i in (1,1,%loop_count%) do (
    REM 这里写上你要执行的命令，注意路径中的空格等需要用引号括起来
    REM "C:\Users\ntbgy\.conda\envs\yao310\python.exe" "movement_logic_yao.py"
    C:\Users\ntbgy\.conda\envs\yao310\python.exe movement_logic_yao.py
    echo 第%%i次执行完成
)

endlocal