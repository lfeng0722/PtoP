# replay_buffer.py
# Lightweight near-miss sample buffer for distribution-based sampling and statistics.

import random

class NearMissReplay:
    """
    Fixed-capacity FIFO buffer storing near-miss scenario records.

    Each record is a dict with keys: "ds", "dd", "dyaw", "F".
    Capacity is enforced by evicting the oldest entry when the buffer is full.
    """

    def __init__(self, capacity=50000):
        self.capacity = int(capacity)
        self.data = []  # list of dict: {"ds", "dd", "dyaw", "F"}

    def __len__(self):
        return len(self.data)

    def add_many(self, items):
        for it in items:
            self.data.append(dict(it))
            if len(self.data) > self.capacity:
                self.data.pop(0)

    def sample(self, k):
        k = min(int(k), len(self.data))
        if k <= 0:
            return []
        return random.sample(self.data, k)
