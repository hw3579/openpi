#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED, ZIP64_VERSION

def is_executable(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def git_list_files(root: str) -> list[str]:
    # 列出：已追踪文件 + 未忽略的未追踪文件（排除 .gitignore/.git/info/exclude 中忽略的）
    cmd = ["git", "-C", os.path.abspath(root),
           "ls-files", "-z", "--cached", "--others", "--exclude-standard", "."]
    out = subprocess.check_output(cmd)
    rels = [p for p in out.decode("utf-8", errors="ignore").split("\x00") if p]
    return rels

def pathspec_list_files(root: str) -> list[str]:
    try:
        import pathspec
    except ImportError as e:
        print("需要 pathspec：pip install pathspec  或者改用 --use-git（若已安装 git）", file=sys.stderr)
        raise

    gi_paths = []
    # 收集所有 .gitignore（根目录 + 子目录）
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames:
            # 仍然遍历子目录，但是我们会手动忽略 .git
            pass
        gi = os.path.join(dirpath, ".gitignore")
        if os.path.isfile(gi):
            gi_paths.append(gi)

    patterns = []
    for gi in gi_paths:
        with open(gi, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # pathspec 的 GitWildMatchPattern 支持 .gitignore 语法
        patterns.append(content)

    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, "\n".join(patterns).splitlines())

    files = []
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root_abs):
        # 总是忽略 .git 目录
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            abs_path = os.path.join(dirpath, name)
            rel = os.path.relpath(abs_path, root_abs)
            # pathspec 使用 POSIX 分隔符判断
            rel_posix = rel.replace(os.sep, "/")
            if spec.match_file(rel_posix):
                continue
            files.append(rel)
    return files

def add_files_to_zip(root: str, rel_files: list[str], zip_path: str, store_no_compress_ext=(".png",".jpg",".jpeg",".gif",".webp",".zip",".7z",".rar",".mp4",".mov",".mkv",".gz",".bz2",".xz",".zst",".pdf")):
    root_abs = os.path.abspath(root)
    # ZIP64 支持大文件
    with ZipFile(zip_path, "w", allowZip64=True) as zf:
        # 选择对已压缩文件（如 mp4/jpg）不再压缩，加快速度
        for rel in rel_files:
            src = os.path.join(root_abs, rel)
            if not os.path.isfile(src):
                # 忽略非常规文件（目录、管道、损坏的链接等）
                continue
            arcname = rel.replace("\\", "/")
            compress_type = ZIP_STORED if os.path.splitext(rel)[1].lower() in store_no_compress_ext else ZIP_DEFLATED
            zf.write(src, arcname=arcname, compress_type=compress_type)

def main():
    ap = argparse.ArgumentParser(description="Zip a folder respecting .gitignore")
    ap.add_argument("folder", help="目标文件夹（工程根目录）")
    ap.add_argument("-o", "--output", help="输出 zip 路径（默认：<folder>.zip）")
    ap.add_argument("--use-git", action="store_true", help="强制使用 git 列表（更精确，需在 Git 仓库中）")
    args = ap.parse_args()

    root = os.path.abspath(args.folder)
    if not os.path.isdir(root):
        print(f"错误：{root} 不是有效的文件夹", file=sys.stderr)
        sys.exit(1)

    out = args.output or (os.path.basename(os.path.normpath(root)) + ".zip")
    out = os.path.abspath(out)

    rel_files = []
    tried_git = False

    if args.use_git or is_executable("git"):
        try:
            rel_files = git_list_files(root)
            tried_git = True
        except Exception as e:
            if args.use_git:
                print("git 列表失败：", e, file=sys.stderr)
                sys.exit(2)
            # 否则继续回退
            rel_files = []

    if not rel_files:
        try:
            rel_files = pathspec_list_files(root)
        except Exception as e:
            # 最后兜底：没有 git 也没有 pathspec，就全部打包（但会包含被忽略文件）
            print("警告：未找到 git 或 pathspec，无法解析 .gitignore，将直接全量打包。", file=sys.stderr)
            rel_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                if ".git" in dirnames:
                    dirnames.remove(".git")
                for name in filenames:
                    abs_path = os.path.join(dirpath, name)
                    rel = os.path.relpath(abs_path, root)
                    rel_files.append(rel)

    print(f"收集文件 {len(rel_files)} 个（来源：{'git' if tried_git and rel_files else 'pathspec' if rel_files else 'raw walk'}）")
    add_files_to_zip(root, rel_files, out)
    print(f"已生成：{out}")

if __name__ == "__main__":
    main()
