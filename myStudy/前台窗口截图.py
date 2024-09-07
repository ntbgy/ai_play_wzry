import pyautogui
import pygetwindow as gw

# 获取特定窗口（这里以窗口标题包含'Notepad'为例，可根据实际情况修改）
target_window = gw.getWindowsWithTitle('LIO-AN00')[0]
# 将窗口移到前台并激活
target_window.activate()

# 获取窗口的位置和大小
x, y, width, height = target_window.left, target_window.top, target_window.width, target_window.height

# 对窗口区域进行截图
screenshot = pyautogui.screenshot(region=(x, y, width, height))
screenshot.save('window_screenshot.png')
