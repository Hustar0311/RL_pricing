"""Versioned joint-control transitions; legacy replay stays unchanged."""
import pickle
import random
from collections import deque, namedtuple

import numpy as np
import torch


JointTransition = namedtuple("JointTransition", "state state2d action next_state next_state2d reward t next_t terminal")


class JointMemory:
    def __init__(self, size, action_dim, device="cpu"):
        self.memory = deque(maxlen=size)
        self.size, self.action_dim, self.device = size, action_dim, device

    def push(self, state, state2d, action, next_state, next_state2d, reward, t, next_t, terminal=False):
        tensors = [v.detach().cpu().clone() for v in [state, state2d, action, next_state, next_state2d]]
        if tensors[2].shape != (1, self.action_dim) or not all(torch.isfinite(v).all() for v in tensors):
            raise ValueError("joint replay contains an invalid action or non-finite state")
        if not np.isfinite(reward) or torch.any(tensors[2].abs() > 1.000001):
            raise ValueError("joint replay requires finite reward and executed normalized actions")
        self.memory.append(JointTransition(*tensors, float(reward), int(t), int(next_t), bool(terminal)))

    def sample(self, batch_size):
        rows = random.sample(list(self.memory), batch_size)
        values = [torch.stack([r[i] for r in rows]) for i in [0, 1]]
        values += [torch.cat([r.action for r in rows]), torch.tensor([r.reward for r in rows]).unsqueeze(1)]
        values += [torch.stack([r[i] for r in rows]) for i in [3, 4]]
        values += [torch.tensor([r[i] for r in rows], dtype=torch.long).unsqueeze(1) for i in [6, 7]]
        values += [torch.tensor([r.terminal for r in rows], dtype=torch.float32).unsqueeze(1)]
        return tuple(v.to(self.device) for v in values)

    def clear(self):
        self.memory.clear()

    def forget(self):
        for _ in range(9 * len(self.memory) // 10):
            self.memory.popleft()

    def save(self, folder, epoch):
        with open(folder + '/memory_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump({"version": 1, "action_dim": self.action_dim, "memory": self.memory}, handle)

    def load(self, folder, epoch):
        with open(folder + '/memory_' + str(epoch) + '.pickle', 'rb') as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("action_dim") != self.action_dim:
            raise ValueError("incompatible joint replay checkpoint")
        self.memory = deque(payload["memory"], maxlen=self.size)

    def __len__(self):
        return len(self.memory)
