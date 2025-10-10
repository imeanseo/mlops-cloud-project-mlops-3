from enum import Enum
from src.model.game_item_cf import ItemCF

class CustomEnum(Enum):
    @classmethod
    def names(cls):
        return [member.name for member in list(cls)]

    @classmethod
    def validation(cls, name: str):
        names = [n.lower() for n in cls.names()]
        if name.lower() in names:
            return True
        else:
            raise ValueError(f"Invalid argument. Must be one of {cls.names()}")

class ModelTypes(CustomEnum):
    ITEM_CF = ItemCF
