from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path


def compress_and_upload(
    source_folder: str,
    archive_name: str,
    machine_index: int,
    remote_name: str = "myDrive",
    remote_parent: str = "",
    archive_output_dir: str | None = None,
    delete_archive_after_upload: bool = False,
) -> Path:
    """
    将指定文件夹压缩为 tar.gz，并通过 rclone 上传到 Google Drive。

    上传目标：
        myDrive:machine_{machine_index}/archive_name.tar.gz

    Parameters
    ----------
    source_folder:
        需要压缩的本地文件夹。

    archive_name:
        压缩包名称，可以写 "results" 或 "results.tar.gz"。

    machine_index:
        机器编号，例如 0 对应 machine_0。

    remote_name:
        rclone 中配置的 remote 名称，例如 myDrive。

    remote_parent:
        Drive 上的可选父目录。
        例如填写 "ExperimentResults"，最终上传到：
        myDrive:ExperimentResults/machine_0/

    archive_output_dir:
        压缩包临时保存目录。
        默认保存在 source_folder 的父目录。

    delete_archive_after_upload:
        上传成功后是否删除本地压缩包。

    Returns
    -------
    Path
        本地压缩包路径。若上传后删除，该路径将不再存在。
    """
    source_path = Path(source_folder).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"文件夹不存在：{source_path}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{source_path}")

    # 自动补充 .tar.gz 后缀
    if archive_name.endswith(".tar.gz"):
        final_archive_name = archive_name
    else:
        final_archive_name = f"{archive_name}.tar.gz"

    if archive_output_dir is None:
        output_dir = source_path.parent
    else:
        output_dir = Path(archive_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / final_archive_name

    print(f"正在压缩：{source_path}")
    print(f"压缩包路径：{archive_path}")

    # 压缩整个文件夹，并保留最外层文件夹名称
    with tarfile.open(archive_path, mode="w:gz") as tar:
        tar.add(
            source_path,
            arcname=source_path.name,
        )

    print(
        f"压缩完成，大小："
        f"{archive_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    # 构造云端目录
    remote_parts = []

    if remote_parent:
        remote_parts.append(remote_parent.strip("/"))

    remote_parts.append(f"machine_{machine_index}")

    remote_folder = "/".join(remote_parts)
    destination = f"{remote_name}:{remote_folder}"

    command = [
        "rclone",
        "copy",
        str(archive_path),
        destination,
        "--ignore-times",
        "--progress",
        "--stats",
        "10s",
        "--retries",
        "10",
        "--low-level-retries",
        "20",
    ]

    print(f"开始上传到：{destination}")

    try:
        subprocess.run(
            command,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "系统中找不到 rclone，请先安装并完成 rclone 配置。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"rclone 上传失败，退出代码：{exc.returncode}"
        ) from exc

    print("上传完成")

    if delete_archive_after_upload:
        archive_path.unlink()
        print(f"已删除本地压缩包：{archive_path}")

    return archive_path
