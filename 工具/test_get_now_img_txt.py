import time

from airtest.cli.parser import cli_setup
from airtest.core.api import auto_setup

from common.airtestProjectsCommon import get_now_img_txt
if not cli_setup():
    auto_setup(
        __file__,
        logdir=True,
        devices=["android:///"]
    )
start = time.time()
txt = get_now_img_txt()
end = time.time()
print(end - start)
# 约7秒