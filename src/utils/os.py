import os
from pathlib import Path
from typing import Literal


def get_num_lines(file_name: str) -> int:
    """
    Efficiently pre-computes the total number of lines contained in an ascii file.

    Source:
        https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python/68385697#68385697
    """

    def _make_gen(reader):
        while True:
            b = reader(2 ** 16)
            if not b: break
            yield b

    with open(file_name, "rb") as f:
        count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))

    return count


def ext_is_in_dir(extension: str, directory: str, mode: Literal["any", "all"] = "any") -> bool:
    """Checks that the files in the given directory have the given extension."""
    elements_end_with_extension = [el.endswith(extension) for el in os.listdir(directory)]
    if mode == "any":
        return any(elements_end_with_extension)
    elif mode == "all":
        return all(elements_end_with_extension)
    else:
        raise ValueError(f"mode {mode} not supported")


def stem_basename_suffix(path: str) -> str:
    basename = get_basename(path)
    return basename[:basename.find(".")]


def get_basename(path: str) -> str:
    return os.path.basename(os.path.normpath(path))
