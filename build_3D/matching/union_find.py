from typing import List, Dict, Any, Hashable

class UnionFind:
    def __init__(self, elements: List[Hashable]):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x: Hashable) -> Hashable:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: Hashable, y: Hashable) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False

    def get_components(self) -> List[List[Hashable]]:
        components = {}
        for el in self.parent:
            root = self.find(el)
            if root not in components:
                components[root] = []
            components[root].append(el)
        return list(components.values())
