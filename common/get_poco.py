from poco.drivers.android.uiautomation import AndroidUiautomationPoco

def get_poco():
    poco = AndroidUiautomationPoco(use_airtest_input=True, screenshot_each_action=False)
    return poco