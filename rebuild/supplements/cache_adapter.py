"""Task-local alternative source-position policy. Original cache code is unchanged."""
from contextlib import contextmanager
import math
import torch
from common import digest
from kvcache import BartCache
import kvgeneration


def random_positions(tokens, ratio, policy_seed):
    if not 0 < ratio <= 1:
        raise ValueError('ratio outside (0,1]')
    length = len(tokens)
    count = max(1, math.ceil(length * ratio))
    if count == length:
        return list(range(length))
    if count == 1:
        return [0]
    seed = int(digest({'source_input_ids': tokens, 'policy_seed': policy_seed})[:15], 16)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    middle = (torch.randperm(length - 2, generator=rng)[:count - 2] + 1).tolist()
    positions = sorted([0, length - 1, *middle])
    assert len(positions) == count and len(set(positions)) == count
    return positions


def random_cache_type(policy_seed):
    class RandomSourceCache(BartCache):
        def initialize(self, inputs):
            # Base constructor sets valid_lengths before calling this method.
            # Change only source-position selection; projection, physical storage,
            # padding masks, absolute positions and eviction remain the base code.
            self.source_positions = [
                random_positions(inputs.input_ids[i, :n].cpu().tolist(), self.cross_ratio, policy_seed)
                for i, n in enumerate(self.valid_lengths)
            ]
            self.cross_width = max(map(len, self.source_positions))
            super().initialize(inputs)

        def metadata(self):
            return {**super().metadata(), 'cross_selection': 'random_endpoints',
                    'selection_seed': policy_seed,
                    'selection_rng_input': 'full model-visible source token IDs plus fixed policy seed'}
    return RandomSourceCache


@contextmanager
def selection_policy(name='evenly_spaced', policy_seed=202611041):
    original = kvgeneration.BartCache
    if name == 'random_endpoints':
        kvgeneration.BartCache = random_cache_type(policy_seed)
    elif name != 'evenly_spaced':
        raise ValueError(f'Unknown source selection policy: {name}')
    try:
        yield
    finally:
        kvgeneration.BartCache = original
