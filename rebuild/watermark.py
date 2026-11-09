"""Kirchenbauer soft green-list, simple_1. CPU PRF is deliberate and recorded.
Seed/key formula and randperm partition match the upstream CPU implementation.
Reference: jwkirchenbauer/lm-watermarking @ 82922516930c02f8aa322765defdb5863d07a00e.
Token scoring ignores specials, initial unknown-prefix token, and repeated bigrams.
"""
from collections import OrderedDict
import math
import torch
from scipy.stats import norm


class Greenlist:
    def __init__(self, vocab_size, gamma=0.25, key=15485863, capacity=1024):
        if not 0 < gamma < 1:
            raise ValueError('gamma must lie in (0,1)')
        self.vocab_size, self.gamma, self.key = vocab_size, gamma, key
        self.capacity=capacity
        self.cache=OrderedDict()
        self.rng=torch.Generator(device='cpu')

    def mask(self, previous):
        previous=int(previous)
        if previous not in self.cache:
            self.rng.manual_seed(self.key*previous)
            ids=torch.randperm(self.vocab_size, generator=self.rng)[:int(self.vocab_size*self.gamma)]
            mask=torch.zeros(self.vocab_size,dtype=torch.bool)
            mask[ids]=True
            self.cache[previous]=mask
            if len(self.cache)>self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(previous)
        return self.cache[previous]

    def batch(self, previous, device):
        # One GPU->CPU transfer per step, not per vocabulary token.
        return torch.stack([self.mask(v) for v in previous.detach().cpu().tolist()]).to(device,non_blocking=True)

    def green(self, previous, token):
        return bool(self.mask(previous)[int(token)])


def scored_positions(sequence, special_ids, eos_id, decoder_start_id):
    """Return raw positions. EOS at index zero is BART's start token, not termination.
    Never join across an internal special token, nor score forced specials.
    Initial lexical token is excluded (portable text-only detector has no prefix).
    """
    sequence=list(sequence)
    begin=1 if sequence and sequence[0]==decoder_start_id else 0
    result=[]; previous_lexical=False
    for pos in range(begin,len(sequence)):
        token=sequence[pos]
        if token==eos_id:
            break
        if token in special_ids:
            previous_lexical=False
            continue
        if previous_lexical:
            result.append(pos)
        previous_lexical=True
    return result


def score(sequence, greenlist, positions, unique=True):
    seen=set(); n=hits=0; included=[]
    for pos in positions:
        pair=(sequence[pos-1],sequence[pos])
        if unique and pair in seen:
            continue
        seen.add(pair); n+=1; hits+=greenlist.green(*pair); included.append(pos)
    if not n:
        return {'n':0,'green':0,'green_fraction':None,'z':None,'p_normal_approx':None,'positions':[]}
    z=(hits-greenlist.gamma*n)/math.sqrt(n*greenlist.gamma*(1-greenlist.gamma))
    return {'n':n,'green':hits,'green_fraction':hits/n,'z':z,
            'p_normal_approx':float(norm.sf(z)),'positions':included}


def clean_content(sequence, special_ids, eos_id, decoder_start_id):
    begin=1 if sequence and sequence[0]==decoder_start_id else 0
    content=[]; raw_positions=[]
    for i in range(begin,len(sequence)):
        if sequence[i]==eos_id:
            break
        if sequence[i] not in special_ids:
            content.append(sequence[i]);raw_positions.append(i)
    return content,raw_positions
