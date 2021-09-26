from abc import ABC, abstractmethod


class Area(ABC):
    def __init__(self): pass


class GrassFireMethod(ABC):
    def __init__(self, matrix, area: Area, fireMethod: str):
        self.matrix = matrix
        self.areas = []
        self.fireMethod = fireMethod

        self.grassFireMethod(area)

    @abstractmethod
    def handle(self, area, grass, x, y):
        pass  # Handling single grass and area

    @abstractmethod
    def condition(self, area, grass) -> bool:
        pass  # Condition for grass to catch on fire

    def grassFire(self, grass, y, x, area, row):
        # Handle the area and single grass. Method made in subclass
        area, grass = self.handle(area, grass, x, y)

        # Iterate all grass next to
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Ignore depending on "fireMethod" Above or all
                if (abs(i) ^ abs(j) and self.fireMethod == '+') or self.fireMethod == '0':
                    # Check grass outside matrix
                    if 0 <= x + i and x + i < len(self.matrix[0]) and 0 <= y + j and y + j < len(self.matrix):
                        grass = self.matrix[y + j][x + i]
                        # Get condition from subclass
                        if self.condition(area, grass):
                            # Continue iterating
                            self.grassFire(grass, y + j, x + i, area, row)

    def grassFireMethod(self, Area: Area):
        for y, row in enumerate(self.matrix):
            for x, grass in enumerate(row):
                area = Area(grass, x, y)
                if self.condition(area, grass):
                    self.grassFire(grass, y, x, area, row)
                    self.areas.append(area)

        return self.areas
