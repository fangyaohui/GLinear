import logging
import os
from logging.handlers import RotatingFileHandler


class ProjectLogger:
    """可复用的项目日志工具类"""

    _configured = False  # 类属性，确保全局只配置一次

    @classmethod
    def configure_logger(cls,
                         log_file: str,
                         level: int = logging.INFO,
                         fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                         max_bytes: int = 10 * 1024 * 1024,  # 10MB
                         backup_count: int = 5):
        """
        配置全局日志系统

        Args:
            log_file: 日志文件完整路径
            level: 日志级别，默认INFO
            fmt: 日志格式
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的历史日志文件数
        """
        if cls._configured:
            return  # 避免重复配置

        # 创建日志目录
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)

        # 主格式化器
        formatter = logging.Formatter(fmt)

        # 文件处理器（带轮转）
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        cls._configured = True

    @staticmethod
    def get_logger(name: str = None) -> logging.Logger:
        """
        获取指定名称的日志器
        Args:
            name: 模块名称（推荐使用 __name__）
        """
        return logging.getLogger(name)