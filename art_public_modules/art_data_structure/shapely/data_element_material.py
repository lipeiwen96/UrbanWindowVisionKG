from dataclasses import dataclass, field
from typing import AnyStr, List, Tuple


@dataclass(order=True)
class DataElementMaterial:
    """
    data element的材质目录
    """
    # TODO: 材质目录的具体是用暂未开发
    # diffuse: AnyStr = field(default='#ffffff')
    # diffuse_opacity: int = field(default=1)
    # is_outline: AnyStr = field(default='false')
    # outline_color: AnyStr = field(default='#bdbdbd')
    # outline_opacity: int = field(default=1)
    # outline_width: int = field(default=1)
    # outline_type: AnyStr = field(default=None)
    # texture_pic: AnyStr = field(default=None)
    # texture_scale: int = field(default=1)
    # texture_rotate: int = field(default=0)
    # effect_type: AnyStr = field(default=None)
    color: AnyStr = field(default='#ffffff')
    opacity: float = field(default=1)
    outline_type: AnyStr = field(default="default")  # "default" "dashed"
