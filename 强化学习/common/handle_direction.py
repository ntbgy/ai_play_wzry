def handle_direction(W_key_pressed, S_key_pressed, A_key_pressed, D_key_pressed):
    """
    根据给定的按键状态确定角色的移动方向。

    参数：
        W_key_pressed (bool): W 键是否被按下。
        S_key_pressed (bool): S 键是否被按下。
        A_key_pressed (bool): A 键是否被按下。
        D_key_pressed (bool): D 键是否被按下。

    返回：
        str: 角色的移动方向，可以是 'up'、'down'、'left'、'right'、'up_left'、'up_right'、'down_left'、'down_right' 中的一个，或者是空字符串表示没有方向。
    """
    # 定义一个字典，将按键组合映射到方向
    direction_map = {
        (True, False, False, False): '上移',
        (False, True, False, False): '下移',
        (False, False, True, False): '左移',
        (False, False, False, True): '右移',
        (True, False, True, False): '左上移',
        (True, False, False, True): '右上移',
        (False, True, True, False): '左下移',
        (False, True, False, True): '右下移'
    }
    # 获取按键状态
    key_state = (W_key_pressed, S_key_pressed, A_key_pressed, D_key_pressed)
    # 如果按键状态在字典中有对应的方向，则返回该方向
    if key_state in direction_map:
        return direction_map[key_state]
    # 如果没有对应的方向，则返回空字符串
    else:
        return ''
