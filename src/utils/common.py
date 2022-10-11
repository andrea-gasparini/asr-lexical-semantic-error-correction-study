from typing import List, TypeVar, Dict

T = TypeVar("T")


def dict_to_list(dict_of_lists: Dict[str, List[T]]) -> List[Dict[str, T]]:
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def list_to_dict(list_of_dicts: List[Dict[str, T]]) -> Dict[str, List[T]]:
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
