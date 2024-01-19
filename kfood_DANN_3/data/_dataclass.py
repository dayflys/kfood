from dataclasses import dataclass

@dataclass
class Food_TrainItem:
    path: str
    food: str
    label: int
    domain: int

@dataclass
class Food_TestItem:
    key: str
    path: int
    domain: int
