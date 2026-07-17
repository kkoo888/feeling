"""工具模块：序列化、指标计算、响应解析等"""

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger("evolution")


def copytree(src: Path, dst: Path, use_symlinks=True):
    """
    将 src 的内容复制到 dst。与 shutil.copytree 不同，dst 目录可以已存在。
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            (dest_f).symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path):
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def extract_archives(path: Path):
    """解压 path 下所有 .zip 文件"""
    for zip_f in path.rglob("*.zip"):
        f_out_dir = zip_f.with_suffix("")

        if f_out_dir.exists():
            logger.debug(
                f"Skipping {zip_f} as an item with the same name already exists."
            )
            if f_out_dir.is_file() and f_out_dir.suffix != "":
                zip_f.unlink()
            continue

        logger.debug(f"Extracting: {zip_f}")
        f_out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_f, "r") as zip_ref:
            zip_ref.extractall(f_out_dir)

        clean_up_dataset(f_out_dir)

        contents = list(f_out_dir.iterdir())

        if len(contents) == 1 and contents[0].name == f_out_dir.name:
            sub_item = contents[0]
            if sub_item.is_dir():
                logger.debug(f"Special handling (child is dir) enabled for: {zip_f}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, f_out_dir)
                sub_item.rmdir()
            elif sub_item.is_file():
                logger.debug(f"Special handling (child is file) enabled for: {zip_f}")
                sub_item_tmp = sub_item.rename(f_out_dir.with_suffix(".__tmp_rename"))
                f_out_dir.rmdir()
                sub_item_tmp.rename(f_out_dir)

        zip_f.unlink()


def preproc_data(path: Path):
    extract_archives(path)
    clean_up_dataset(path)
