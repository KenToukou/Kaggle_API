import logging
import os
from datetime import datetime
from pathlib import Path


def configure_logging(folder_name):
    """
    ロギングを設定します。
    """
    try:
        current_dir = os.getcwd()
        output_log_folder = os.path.join(current_dir, "log", folder_name)
        if not Path(output_log_folder).exists():
            Path(output_log_folder).mkdir(parents=True)

        log_file_name = f"{output_log_folder}/log.txt"  # noqa
        # 既存のハンドラをクリアします。
        logging.getLogger().handlers = []

        # ハンドラなしで基本設定をリセットします。
        logging.basicConfig(handlers=[])

        # ファイルハンドラを設定します。
        file_handler = logging.FileHandler(log_file_name, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )  # メッセージのみをフォーマットします。
        logging.getLogger().addHandler(file_handler)

        # コンソールハンドラを同じフォーマットで設定します。
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )  # コンソール出力もメッセージのみにします。
        logging.getLogger().addHandler(console_handler)

        # グローバルなログレベルをINFOに設定します。
        logging.getLogger().setLevel(logging.INFO)

        print(f"ログファイル出力：{log_file_name}")
    except Exception as e:
        print(f"Failed to configure logging: {e}")


def configure_logging_score(folder_name):
    """
    ロギングを設定します。
    """
    try:
        current_dir = os.getcwd()
        output_log_folder = os.path.join(current_dir, "log", folder_name)
        if not Path(output_log_folder).exists():
            Path(output_log_folder).mkdir(parents=True)

        log_file_name = f"{output_log_folder}/score{datetime.now().strftime('%Y年%m月%d日_%H時%M分%S秒')}.txt"  # noqa
        # 既存のハンドラをクリアします。
        logging.getLogger().handlers = []

        # ハンドラなしで基本設定をリセットします。
        logging.basicConfig(handlers=[])

        # ファイルハンドラを設定します。
        file_handler = logging.FileHandler(log_file_name, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(message)s")
        )  # メッセージのみをフォーマットします。
        logging.getLogger().addHandler(file_handler)

        # コンソールハンドラを同じフォーマットで設定します。
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )  # コンソール出力もメッセージのみにします。
        logging.getLogger().addHandler(console_handler)

        # グローバルなログレベルをINFOに設定します。
        logging.getLogger().setLevel(logging.INFO)

        print(f"ログファイル出力：{log_file_name}")
    except Exception as e:
        print(f"Failed to configure logging: {e}")
