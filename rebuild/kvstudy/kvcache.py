"""Physical BART tuple-cache compaction with absolute decoder positions preserved.
Applies only to this model instance; no library files or shared workers are patched.
Encoder always sees the full input. Cross-cache downsampling is not source truncation.
"""
import math
import types
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


def evenly_spaced(length,ratio):
    if not 0<ratio<=1:raise ValueError('Cache ratio outside (0,1]')
    count=max(1,math.ceil(length*ratio))
    indices=torch.linspace(0,length-1,count,dtype=torch.float64).round().long()
    assert indices.unique().numel()==count
    return indices


def tensor_bytes(t):return t.numel()*t.element_size()


class BartCache:
    def __init__(self,model,inputs,cross_ratio=1.0,self_ratio=1.0):
        self.model=model;self.cross_ratio=cross_ratio;self.self_ratio=self_ratio
        if not 0<self_ratio<=1:raise ValueError('Invalid self ratio')
        self.device=model.device;self.batch=inputs.input_ids.shape[0]
        self.layers=model.model.decoder.layers
        self.cross_calls=[{'k':0,'v':0} for _ in self.layers]
        self.handles=[];self.absolute_position=0;self.observed_position=None
        self.past=None;self.retained_self=[];self.history=[]
        self.position_module=model.model.decoder.embed_positions
        self.original_forward=self.position_module.forward
        self.active=False
        self.source_width=int(inputs.input_ids.shape[1])
        self.valid_lengths=inputs.attention_mask.sum(-1).cpu().tolist()
        self.source_positions=[evenly_spaced(n,cross_ratio).tolist() for n in self.valid_lengths]
        self.cross_width=max(map(len,self.source_positions))
        try:
            self.initialize(inputs)
        except Exception:
            # Constructor OOM must not leave projection hooks attached to a reused model.
            for handle in self.handles:handle.remove()
            self.handles.clear()
            raise

    def initialize(self,inputs):
        full=self.model.get_encoder()(**inputs,return_dict=True).last_hidden_state
        index=torch.zeros((self.batch,self.cross_width),device=self.device,dtype=torch.long)
        mask=torch.zeros_like(index)
        for i,positions in enumerate(self.source_positions):
            index[i,:len(positions)]=torch.tensor(positions,device=self.device)
            mask[i,:len(positions)]=1
        # This encoder-output view only supplies matching shape to HF's cached
        # cross-attention path. It was computed with FULL encoder context.
        reduced=full.gather(1,index.unsqueeze(-1).expand(-1,-1,full.shape[-1])).contiguous()
        self.encoder=BaseModelOutput(last_hidden_state=reduced)
        self.mask=mask
        past=[]
        for li,layer in enumerate(self.layers):
            attention=layer.encoder_attn
            def counter(which,idx):
                def hook(module,args,result):self.cross_calls[idx][which]+=1
                return hook
            self.handles.append(attention.k_proj.register_forward_hook(counter('k',li)))
            self.handles.append(attention.v_proj.register_forward_hook(counter('v',li)))
            full_k=attention._shape(attention.k_proj(full),-1,self.batch)
            full_v=attention._shape(attention.v_proj(full),-1,self.batch)
            gather=index[:,None,:,None].expand(self.batch,attention.num_heads,self.cross_width,attention.head_dim)
            k=full_k.gather(2,gather).contiguous();v=full_v.gather(2,gather).contiguous()
            # Assert physical storage shrank; a view holding the full cache is insufficient.
            assert k.untyped_storage().nbytes()==tensor_bytes(k)
            assert v.untyped_storage().nbytes()==tensor_bytes(v)
            empty=k.new_empty(self.batch,attention.num_heads,0,attention.head_dim)
            past.append((empty,empty.clone(),k,v))
            del full_k,full_v
        self.past=tuple(past)
        self.cross_ptrs=[(p[2].data_ptr(),p[3].data_ptr()) for p in self.past]
        self.cross_bytes=sum(tensor_bytes(t) for p in self.past for t in p[2:])
        self.full_cross_bytes=self.cross_bytes*self.source_width//self.cross_width
        del full,reduced

    def __enter__(self):
        if self.active:raise RuntimeError('Cache context re-entered')
        controller=self
        def positions(module,input_ids,past_key_values_length=0):
            # Do not derive absolute positions from the physically shortened past.
            controller.observed_position=controller.absolute_position
            return controller.original_forward(input_ids,past_key_values_length=controller.absolute_position)
        self.position_module.forward=types.MethodType(positions,self.position_module)
        self.active=True
        return self

    def __exit__(self,exc_type,exc,tb):
        self.position_module.forward=self.original_forward
        for handle in self.handles:handle.remove()
        self.handles.clear();self.active=False

    def forward(self,previous,absolute_step):
        if not self.active:raise RuntimeError('Use cache as a context manager')
        if absolute_step!=len(self.history):raise ValueError('Decoder input step is not sequential')
        self.absolute_position=absolute_step;self.observed_position=None
        prior_length=len(self.retained_self)
        output=self.model(encoder_outputs=self.encoder,attention_mask=self.mask,
                          decoder_input_ids=previous,past_key_values=self.past,
                          use_cache=True,return_dict=True)
        if self.observed_position!=absolute_step:raise AssertionError('Absolute position override was not used')
        updated=output.past_key_values
        positions=self.retained_self+[absolute_step]
        budget=max(1,math.ceil((absolute_step+1)*self.self_ratio))
        if len(positions)>budget:
            slots=[0]+list(range(len(positions)-(budget-1),len(positions))) if budget>1 else [0]
        else:slots=list(range(len(positions)))
        assert len(slots)==min(len(positions),budget)
        ids=torch.tensor(slots,device=self.device,dtype=torch.long)
        retained=[positions[s] for s in slots]
        new_past=[]
        for li,p in enumerate(updated):
            if len(p)!=4:raise AssertionError('Expected BART (selfK,selfV,crossK,crossV)')
            assert p[0].shape[2]==prior_length+1 and p[1].shape[2]==prior_length+1
            assert p[2].shape[2]==self.cross_width and p[3].shape[2]==self.cross_width
            assert (p[2].data_ptr(),p[3].data_ptr())==self.cross_ptrs[li], 'Cross KV was recomputed'
            assert self.cross_calls[li]=={'k':1,'v':1}, 'Cross KV projections executed during decoding'
            if len(slots)<len(positions):
                sk=p[0].index_select(2,ids).contiguous();sv=p[1].index_select(2,ids).contiguous()
            else:sk,sv=p[:2]
            assert sk.shape[2]==len(retained) and sv.shape[2]==len(retained)
            assert sk.untyped_storage().nbytes()==tensor_bytes(sk)
            assert sv.untyped_storage().nbytes()==tensor_bytes(sv)
            new_past.append((sk,sv,p[2],p[3]))
        self.past=tuple(new_past);self.retained_self=retained
        self_bytes=sum(tensor_bytes(t) for p in self.past for t in p[:2])
        full_self_bytes=self_bytes*(absolute_step+1)//len(retained)
        audit={'decoder_absolute_position':absolute_step,'positions_seen':absolute_step+1,
               'self_cache_before_forward':prior_length,'self_cache_after_append':prior_length+1,
               'self_cache_after_eviction':len(retained),'retained_self_positions':retained,
               'cross_padded_width':self.cross_width,
               'cross_KV_bytes_batch':self.cross_bytes,'self_KV_bytes_batch':self_bytes,
               'uncompressed_cross_KV_bytes_batch':self.full_cross_bytes,
               'uncompressed_self_KV_bytes_batch':full_self_bytes,
               'KV_bytes_batch':self.cross_bytes+self_bytes,
               'full_KV_bytes_batch_at_same_step':self.full_cross_bytes+full_self_bytes,
               'cross_cache_recomputed':False}
        self.history.append(audit)
        return output.logits[:,-1,:].float(),audit

    def metadata(self):
        return {'cross_ratio':self.cross_ratio,'self_ratio':self.self_ratio,
                'cross_valid_source_lengths':self.valid_lengths,
                'cross_retained_source_positions':self.source_positions,
                'cross_original_padded_width':self.source_width,'cross_retained_padded_width':self.cross_width,
                'cross_projection_calls_per_layer':self.cross_calls,
                'cache_by_decoder_step':self.history,
                'absolute_positions_preserved':True,'full_source_encoded_before_cache_eviction':True}
