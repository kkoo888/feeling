"""序列化工具：将进化数据结构序列化/反序列化为 JSON"""

import copy
import json
from pathlib import Path
from typing import Type, TypeVar

import dataclasses_json
from ..journal import EvolutionJournal


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """将进化数据结构序列化为 JSON 字符串"""
    if isinstance(obj, EvolutionJournal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, EvolutionJournal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    """将进化数据结构写入 JSON 文件"""
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """从 JSON 字符串反序列化进化数据结构"""
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, EvolutionJournal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    """从 JSON 文件反序列化进化数据结构"""
    with open(path, "r") as f:
        return loads_json(f.read(), cls)
