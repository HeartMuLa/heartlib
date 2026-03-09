import argparse
import os
import sys
import logging
import importlib.util
from huggingface_hub import snapshot_download, hf_hub_download


# UI & Logging (Same as before)
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan
        + "%(asctime)s"
        + reset
        + " ["
        + green
        + "%(levelname)s"
        + reset
        + "] %(message)s",
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt, datefmt="%H:%M:%S").format(record)


logger = logging.getLogger("Downloader")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)


def print_banner():
    print(f"\033[1;32m")
    print(r"""
  _    _                 _   __  __       _      
 | |  | |               | | |  \/  |     | |     
 | |__| | ___  __ _ _ __| |_| \  / |_   _| | __ _ 
 |  __  |/ _ \/ _` | '__| __| |\/| | | | | |/ _` |
 | |  | |  __/ (_| | |  | |_| |  | | |_| | | (_| |
 |_|  |_|\___|\__,_|_|   \__|_|  |_|\__,_|_|\__,_|
            TURBO DOWNLOAD MANAGER v2
    """)
    print(f"\033[0m")


# REPOS
REPO_CODEC = "HeartMuLa/HeartCodec-oss"
REPO_3B = "HeartMuLa/HeartMuLa-oss-3B"
REPO_TRANSCRIPTOR = "HeartMuLa/HeartTranscriptor-oss"
REPO_CONFIGS = "HeartMuLa/HeartMuLaGen"


def enable_speed_hacks(use_mirror):
    if importlib.util.find_spec("hf_transfer"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        logger.info("🚀 \033[1;35mhf_transfer\033[0m enabled!")
    if use_mirror:
        mirror_url = "https://hf-mirror.com" if use_mirror is True else use_mirror
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"🌐 Using Mirror: {mirror_url}")


def download_repo(repo_id, local_folder, target_dir, workers):
    path = os.path.join(target_dir, local_folder)
    logger.info(f"Downloading {local_folder}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*"],
        max_workers=workers,
    )
    logger.info(f"{local_folder} complete.")


def download_configs(target_dir):
    logger.info(f"Downloading Configs...")
    for filename in ["tokenizer.json", "gen_config.json"]:
        hf_hub_download(
            repo_id=REPO_CONFIGS,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )


def main():
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./ckpt")
    parser.add_argument("--mirror", nargs="?", const=True, default=False)
    parser.add_argument("--workers", type=int, default=8)

    group = parser.add_argument_group("Selection")
    group.add_argument("--all", action="store_true")
    group.add_argument("--codec", action="store_true")
    group.add_argument("--model3b", action="store_true")
    group.add_argument(
        "--transcriptor", action="store_true", help="Download HeartTranscriptor"
    )
    group.add_argument("--configs", action="store_true")

    args = parser.parse_args()
    enable_speed_hacks(args.mirror)

    if not (args.codec or args.model3b or args.configs or args.transcriptor):
        args.all = True

    target_dir = os.path.abspath(args.dir)
    os.makedirs(target_dir, exist_ok=True)

    try:
        if args.all or args.configs:
            download_configs(target_dir)
        if args.all or args.codec:
            download_repo(REPO_CODEC, "HeartCodec-oss", target_dir, args.workers)
        if args.all or args.model3b:
            download_repo(REPO_3B, "HeartMuLa-oss-3B", target_dir, args.workers)
        if args.all or args.transcriptor:
            download_repo(
                REPO_TRANSCRIPTOR, "HeartTranscriptor-oss", target_dir, args.workers
            )

        logger.info("\033[1;32mDownloads finished!\033[0m")
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
