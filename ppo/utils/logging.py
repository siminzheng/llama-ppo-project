import warnings

def disable_warnings():
    """
    全局屏蔽 Python 的所有警告信息。

    主要用于训练过程中，防止大量模型加载、量化、分词等非关键性警告
    干扰控制台输出，保持日志干净。
    """
    warnings.filterwarnings("ignore")
