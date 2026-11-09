"""Engineering tests for real physical cache intervention, not paper evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import unittest
from copy import deepcopy
import torch
from transformers import BartConfig,BartForConditionalGeneration,AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from common import ROOT,load_protocol,atomic_json
from kvcache import BartCache,evenly_spaced


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(55)
        c=BartConfig(vocab_size=80,d_model=32,encoder_layers=1,decoder_layers=2,
                     encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=64,decoder_ffn_dim=64)
        c._attn_implementation='sdpa'
        self.model=BartForConditionalGeneration(c).eval()
        self.inputs=BatchEncoding({'input_ids':torch.tensor([[0,10,11,12,13,14,15,2],[0,20,21,2,1,1,1,1]]),
                                  'attention_mask':torch.tensor([[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0]])})
        self.prefix=torch.tensor([[2,0,30,31,32,33,34,35],[2,0,40,41,42,43,44,45]])

    def reference(self):
        with torch.no_grad():
            enc=self.model.get_encoder()(**self.inputs,return_dict=True)
            return [self.model(encoder_outputs=enc,attention_mask=self.inputs.attention_mask,
                               decoder_input_ids=self.prefix[:,:i+1],use_cache=False).logits[:,-1,:].clone()
                    for i in range(self.prefix.shape[1])]

    def test_full_cache_exact_logits_and_shape(self):
        expected=self.reference();original=self.model.model.decoder.embed_positions.forward
        with torch.no_grad(),BartCache(self.model,self.inputs,1,1) as cache:
            for step,reference in enumerate(expected):
                logits,a=cache.forward(self.prefix[:,step:step+1],step)
                self.assertTrue(torch.allclose(logits,reference,atol=1e-5))
                self.assertEqual(a['KV_bytes_batch'],a['full_KV_bytes_batch_at_same_step'])
                self.assertEqual(a['retained_self_positions'],list(range(step+1)))
        self.assertEqual(self.model.model.decoder.embed_positions.forward,original)

    def test_cross_selection_storage_and_effect(self):
        expected=self.reference()
        with torch.no_grad(),BartCache(self.model,self.inputs,.5,1) as cache:
            self.assertEqual(cache.source_positions,[[0,2,5,7],[0,3]])
            self.assertEqual(cache.mask.tolist(),[[1,1,1,1],[1,1,0,0]])
            effects=[]
            for step,reference in enumerate(expected):
                logits,a=cache.forward(self.prefix[:,step:step+1],step)
                effects.append(float((logits-reference).abs().max()))
                self.assertEqual(a['cross_KV_bytes_batch']*2,a['uncompressed_cross_KV_bytes_batch'])
                self.assertFalse(a['cross_cache_recomputed'])
            self.assertGreater(max(effects),1e-6)
            self.assertTrue(all(c=={'k':1,'v':1} for c in cache.cross_calls))

    def test_self_eviction_keeps_absolute_positions(self):
        expected=self.reference()
        with torch.no_grad(),BartCache(self.model,self.inputs,1,.5) as cache:
            for step in range(8):
                logits,a=cache.forward(self.prefix[:,step:step+1],step)
                self.assertEqual(a['decoder_absolute_position'],step)
                self.assertEqual(cache.observed_position,step)
                self.assertEqual(a['self_cache_after_eviction'],(step+2)//2)
                self.assertEqual(a['retained_self_positions'][0],0)
                self.assertEqual(a['cross_KV_bytes_batch'],a['uncompressed_cross_KV_bytes_batch'])
            self.assertEqual(cache.retained_self,[0,5,6,7])
            self.assertGreater(float((logits-expected[-1]).abs().max()),1e-6)

    def test_both_components_and_cleanup_on_exception(self):
        original=self.model.model.decoder.embed_positions.forward
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with torch.no_grad(),BartCache(self.model,self.inputs,.5,.5) as cache:
                for step in range(8):logits,a=cache.forward(self.prefix[:,step:step+1],step)
                self.assertEqual(a['KV_bytes_batch']*2,a['full_KV_bytes_batch_at_same_step'])
                raise RuntimeError('intentional')
        self.assertEqual(self.model.model.decoder.embed_positions.forward,original)
        self.assertTrue(all(not l.encoder_attn.k_proj._forward_hooks for l in self.model.model.decoder.layers))


def real_checkpoint_check(model,tokenizer,samples,config,out):
    """Matched full-cache sampler + teacher-forced compressed forward evidence."""
    from generation import decode_batch
    from kvgeneration import decode_cache_batch
    cfg=deepcopy(config);cfg['generation']['max_new_tokens']=32;cfg['cache_condition']='full'
    chosen=samples[:2];evidence=[]
    for delta in (0.0,2.0):
        reference=decode_batch(model,tokenizer,chosen,cfg,42,delta)
        actual,audit=decode_cache_batch(model,tokenizer,chosen,cfg,42,delta)
        if [r['raw_ids'] for r in reference]!=[r['raw_ids'] for r in actual]:
            raise AssertionError('Full-cache adapter changes matched native-validated sampling')
        evidence.append({'test':'matched_full_sampling','delta':delta,'exact_token_match':True,'documents':len(chosen)})
    inputs=tokenizer.pad({'input_ids':[s['source_input_ids'] for s in chosen]},padding=True,return_tensors='pt').to(model.device)
    prefix=torch.tensor([r['raw_ids'][:16] for r in reference],device=model.device)
    with torch.inference_mode():
        with BartCache(model,inputs,1,1) as cache:
            full=[cache.forward(prefix[:,i:i+1],i)[0].clone() for i in range(16)]
        for name,cr,sr in [('cross-half',.5,1),('self-half',1,.5),('both-half',.5,.5)]:
            with BartCache(model,inputs,cr,sr) as cache:
                differences=[]
                for i in range(16):
                    logits,last=cache.forward(prefix[:,i:i+1],i)
                    differences.append(float((logits-full[i]).abs().max()))
                if max(differences)<1e-6:raise AssertionError('Compression does not change logits')
                evidence.append({'test':'real_cache_intervention','condition':name,'max_logit_change_fixed_prefix':max(differences),
                                 'last_step':last,'cache':cache.metadata()})
    atomic_json(out/'real_checkpoint_checks.json',{'purpose':'engineering_only','checks':evidence})
    return evidence


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
