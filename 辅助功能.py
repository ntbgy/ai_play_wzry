import numpy as np


def 状态信息综合(图片张量, 操作序列, trg_mask):
    状态 = {'图片张量': 图片张量[np.newaxis, :], '操作序列': 操作序列, 'trg_mask': trg_mask}
    return 状态
