"""Full-sequence correctness graph; pads are excluded from full attention's keys."""
import torch
from ..llm_common.qwen35 import Qwen35Trunk


def block_bias(input_ids, last_pos, prefix_lengths):
    pos = torch.arange(input_ids.shape[1], device=input_ids.device)
    suffix = pos[None, :] >= prefix_lengths[:, None]
    visible = ((~suffix[:, None, :]) | suffix[:, :, None])
    visible = visible & (pos[None, None, :] <= last_pos[:, None, None])
    return torch.where(visible[:, None], 0.0, float("-inf"))


class ElephantGraph(torch.nn.Module):
    def __init__(self, text_model, lm_head, slot_ids):
        super().__init__()
        self.trunk = Qwen35Trunk(text_model)
        self.lm_head = lm_head
        self.register_buffer("slot_ids", torch.tensor(slot_ids), persistent=False)

    def forward(self, input_ids, last_pos, prefix_lengths):
        h = self.trunk(input_ids, block_bias(input_ids, last_pos, prefix_lengths)).float()
        ans = h[torch.arange(h.shape[0], device=h.device), last_pos]
        return ans @ self.lm_head.weight[self.slot_ids].float().T
