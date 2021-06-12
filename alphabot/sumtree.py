from typing import Tuple
from .observation import Observation
import numpy as np

class SumTree(object):
    index = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
    
    def add(self, priority, data) -> None:
        current_index = self.index + self.capacity - 1
        self.data[self.index] = data
        self.index += 1
        self.update(priority, current_index)

        if self.index >= self.capacity:  
            self.index = 0

    def update(self, priority, current_index) -> None:
        delta = priority - self.tree[current_index]
        while current_index > 0:
            current_index = (current_index - 1) // 2
            self.tree[current_index] += delta

    def get_leaf(self, value) -> Tuple[int, float, Observation]:
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_child_index]:
                parent_index = left_child_index

            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self) -> float:
        return self.tree[0]