import logging

import colorama
from colorama import Fore, Style

colorama.init()


class ColoredFormatter(logging.Formatter):
    # my_formatter = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)-8s - %(message)s'
    asctime = Style.DIM + Fore.GREEN + '%(asctime)s'
    decollator = Style.NORMAL + Fore.RED + ' | '
    code_line = Style.DIM + Fore.CYAN + '%(module)s.%(funcName)s:%(lineno)d'
    decollator_2 = Style.NORMAL + Fore.RED + ' - '
    levelname = Style.BRIGHT + Fore.BLUE + '%(levelname)-8s'
    FORMATS = {
        logging.DEBUG: asctime + decollator + levelname + decollator + code_line + decollator_2 + Style.DIM + Fore.WHITE + '%(message)s' + Style.RESET_ALL,
        logging.INFO: asctime + decollator + levelname + decollator + code_line + decollator_2 + Style.NORMAL + Fore.GREEN + '%(message)s' + Style.RESET_ALL,
        logging.WARNING: asctime + decollator + levelname + decollator + code_line + decollator_2 + Style.RESET_ALL + Fore.YELLOW + '%(message)s' + Style.RESET_ALL,
        logging.ERROR: asctime + decollator + levelname + decollator + code_line + decollator_2 + Style.BRIGHT + Fore.RED + '%(message)s' + Style.RESET_ALL,
        logging.CRITICAL: asctime + decollator + levelname + decollator + code_line + decollator_2 + Style.BRIGHT + Fore.RED + '%(message)s' + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# 获取 Logger 对象
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

# # 创建并设置 StreamHandler
# my_formatter = '%(asctime)s - %(filename)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s'
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_formatter = logging.Formatter(my_formatter)
# console_handler.setFormatter(console_formatter)
#
# # 创建并设置 FileHandler
# file_handler = logging.FileHandler('my_log.log')
# file_handler.setLevel(logging.DEBUG)
# file_formatter = logging.Formatter(my_formatter)
# file_handler.setFormatter(file_formatter)
#
# # 将 Handler 添加到 Logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

if __name__ == '__main__':
    # 使用 Logger 记录日志
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
