class stop:
    """
    存储检测游戏是否结束需要的变量
    """

    def __init__(self):
        self.stop = False
        self.image_path = None

    def get_stop(self):
        return self.stop

    def set_stop(self, stop=False):
        self.stop = stop

    def get_image_path(self):
        return self.image_path

    def set_image_path(self, image_path=None):
        self.image_path = image_path