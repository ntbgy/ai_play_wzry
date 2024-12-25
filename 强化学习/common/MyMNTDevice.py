from pyminitouch import MNTDevice


class MyMNTDevice(MNTDevice):
    def __init__(self, ID):
        MNTDevice.__init__(self, ID)

    def 发送(self, 内容):
        self.connection.send(内容)
