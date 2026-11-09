"""Fast correctness checks. Synthetic outputs here are NOT experimental results."""
import ast
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import BartConfig,BartForConditionalGeneration
from common import ROOT,atomic_json,read_json,lock_manifest
from watermark import Greenlist,scored_positions,score,clean_content
from analysis import calibrated_threshold,auc


class Correctness(unittest.TestCase):
    def test_upstream_simple1_cpu_parity(self):
        source=(ROOT/'reference/watermark_processor.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='WatermarkBase')
        ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<upstream>','exec'),ns)
        upstream=ns['WatermarkBase'](vocab=list(range(50264)),gamma=.25,hash_key=15485863)
        upstream.rng=torch.Generator(device='cpu');own=Greenlist(50264)
        for prev in [0,1,2,17,351,40000]:
            ids=upstream._get_greenlist_ids(torch.tensor([prev]));mask=torch.zeros(50264,dtype=torch.bool);mask[ids]=True
            self.assertTrue(torch.equal(mask,own.mask(prev)))

    def test_padding_eos_and_repeat_handling(self):
        gl=Greenlist(100);special={0,1,2}; seq=[2,0,10,11,12,10,11,2]
        pos=scored_positions(seq,special,2,2)
        self.assertEqual(pos,[3,4,5,6]);base=score(seq,gl,pos)
        self.assertEqual(base['n'],3)
        padded=seq+[1]*20+[50,51]
        self.assertEqual(base,score(padded,gl,scored_positions(padded,special,2,2)))
        ids,indices=clean_content(padded,special,2,2)
        self.assertEqual(ids,[10,11,12,10,11]);self.assertEqual(indices,[2,3,4,5,6])
        self.assertEqual(score([2,0,2],gl,scored_positions([2,0,2],special,2,2))['z'],None)

    def test_soft_bias_probability(self):
        gl=Greenlist(512);mask=gl.mask(73)
        base=torch.linspace(-3,3,512,dtype=torch.float64).softmax(-1);delta=2
        q=base[mask].sum();expected=torch.exp(torch.tensor(delta))*q/(1-q+torch.exp(torch.tensor(delta))*q)
        biased=(base.log()+delta*mask).softmax(-1)
        self.assertAlmostEqual(float(expected),float(biased[mask].sum()),places=6)
        self.assertTrue(torch.allclose((base.log()+0*mask).softmax(-1),base))

    def test_cache_matches_full_decoder(self):
        torch.manual_seed(77)
        c=BartConfig(vocab_size=64,d_model=32,encoder_layers=1,decoder_layers=1,
                     encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=64,decoder_ffn_dim=64)
        m=BartForConditionalGeneration(c).eval()
        src=torch.tensor([[0,10,11,2]]); attn=torch.ones_like(src);dec=torch.tensor([[2,0,20,21,22]])
        with torch.no_grad():
            enc=m.get_encoder()(input_ids=src,attention_mask=attn,return_dict=True);past=None
            for i in range(dec.shape[1]):
                cached=m(encoder_outputs=enc,attention_mask=attn,decoder_input_ids=dec[:,i:i+1],past_key_values=past,use_cache=True)
                full=m(encoder_outputs=enc,attention_mask=attn,decoder_input_ids=dec[:,:i+1],use_cache=False)
                self.assertTrue(torch.allclose(cached.logits[:,-1],full.logits[:,-1],atol=1e-5))
                past=cached.past_key_values

    def test_atomic_checkpoint_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d);identity=lock_manifest(p,{'protocol':{'a':1}})
            self.assertEqual(identity,lock_manifest(p,{'protocol':{'a':1}}))
            with self.assertRaises(RuntimeError):lock_manifest(p,{'protocol':{'a':2}})
            atomic_json(p/'batch.json',{'x':[1,2]});self.assertEqual(read_json(p/'batch.json'),{'x':[1,2]})
            self.assertFalse((p/'batch.json.tmp').exists())

    def test_sampling_rng_and_model_vocab_boundary(self):
        from copy import deepcopy
        from transformers import AutoTokenizer
        from generation import decode_batch
        tokenizer_path=ROOT/'assets/bart-large-cnn'
        if not torch.cuda.is_available() or not (tokenizer_path/'tokenizer.json').exists():
            self.skipTest('GPU/tokenizer not yet available; covered by later integration smoke')
        tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        # Tiny RANDOM model tests code paths, not scientific performance.
        model=BartForConditionalGeneration(BartConfig(vocab_size=50264,d_model=32,encoder_layers=1,decoder_layers=1,
                         encoder_attention_heads=2,decoder_attention_heads=2,encoder_ffn_dim=64,decoder_ffn_dim=64)).cuda().eval()
        config=deepcopy(read_json(ROOT/'protocol.json'))
        config['generation'].update(max_new_tokens=6,min_content_tokens=2)
        samples=[{'id':'unit-a','role':'unit','source_input_ids':[0,10,11,2]},
                 {'id':'unit-b','role':'unit','source_input_ids':[0,12,2]}]
        a=decode_batch(model,tok,samples,config,42,0)
        b=decode_batch(model,tok,samples,config,42,0)
        self.assertEqual([r['raw_ids'] for r in a],[r['raw_ids'] for r in b])
        c=decode_batch(model,tok,samples,config,42,2)
        self.assertEqual(len(c),2)
        for r in a+c:
            self.assertTrue(all(t<50264 for t in r['raw_ids']))
            self.assertEqual(r['raw_ids'][:2],[2,0])
            self.assertEqual(len(r['trace']),len(r['raw_ids'])-1)

    def test_nli_pair_and_bart_sentence_alignment(self):
        from transformers import AutoTokenizer
        from evaluation import sentence_units
        if not (ROOT/'assets/nli-deberta-v3-base/tokenizer.json').exists():
            self.skipTest('Tokenizers not yet available; covered by later integration smoke')
        tok=AutoTokenizer.from_pretrained(ROOT/'assets/bart-large-cnn',local_files_only=True)
        nli=AutoTokenizer.from_pretrained(ROOT/'assets/nli-deberta-v3-base',local_files_only=True)
        text='Alice bought three apples. Bob lives in Paris.'
        content=tok(text,add_special_tokens=False)['input_ids'];raw=[2,0]+content+[2]
        row={'summary':text,'content_ids':content,'raw_ids':raw,'content_raw_positions':list(range(2,2+len(content))),
             'trace':[{'raw_index':i,'base_entropy':1.0} for i in range(2,2+len(content))]}
        units=sentence_units(row,{'source_input_ids':content},tok,Greenlist(50264),read_json(ROOT/'protocol.json'))
        self.assertEqual(len(units),2);self.assertTrue(all(u['alignment_valid'] for u in units))
        premise=nli(text,add_special_tokens=False)['input_ids'];hyp=nli('Alice bought three apples.',add_special_tokens=False)['input_ids']
        pair=nli.prepare_for_model(premise,pair_ids=hyp,add_special_tokens=True,return_attention_mask=True,truncation=False)
        batch=nli.pad([pair],return_tensors='pt')
        self.assertEqual(int(batch.attention_mask.sum()),len(pair['input_ids']))
        self.assertLess(len(pair['input_ids']),512)

    def test_noncanonical_BPE_and_invalid_unicode_alignment(self):
        from transformers import AutoTokenizer
        from alignment import original_token_offsets,utf8_char_spans,sentence_assignment
        import random
        rng=random.Random(90)
        for _ in range(100):
            blob=bytes(rng.randrange(256) for _ in range(64))
            text,spans,invalid=utf8_char_spans(blob)
            self.assertEqual(text,blob.decode('utf-8',errors='replace'))
            self.assertEqual(len(text),len(spans))
        if not (ROOT/'assets/bart-large-cnn/tokenizer.json').exists():
            self.skipTest('BART tokenizer not yet available')
        tok=AutoTokenizer.from_pretrained(ROOT/'assets/bart-large-cnn',local_files_only=True)
        ids=[245,2036,4,47856,15264,49171,4]  # noncanonical 5+22, split Chinese UTF-8, invalid bytes
        text=tok.decode(ids,clean_up_tokenization_spaces=False)
        self.assertNotEqual(tok(text,add_special_tokens=False)['input_ids'],ids)
        offsets,invalid=original_token_offsets(tok,ids,text)
        self.assertEqual(len(offsets),len(ids));self.assertGreater(invalid,0)
        self.assertEqual(offsets[0],(0,1));self.assertEqual(offsets[1],(1,3))
        self.assertEqual(offsets[3],offsets[4])  # same character across two byte-piece tokens
        groups=sentence_assignment(offsets,[(0,4),(4,len(text))])
        self.assertEqual(sorted(i for g in groups for i in g),list(range(len(ids))))

    def test_topk_topp_filters(self):
        from generation import sampling_filters
        scores=torch.log(torch.tensor([[.50,.30,.15,.05]],dtype=torch.float64))
        ids=torch.tensor([[2]])
        unchanged=sampling_filters({'top_k':0,'top_p':1.0})(ids,scores.clone())
        self.assertTrue(torch.equal(unchanged,scores))
        k=sampling_filters({'top_k':2,'top_p':1.0})(ids,scores.clone())
        p=sampling_filters({'top_k':0,'top_p':.75})(ids,scores.clone())
        self.assertEqual(torch.isfinite(k).tolist(),[[True,True,False,False]])
        self.assertEqual(torch.isfinite(p).tolist(),[[True,True,False,False]])
        # Bias must act before candidate filtering: promote the originally lowest token.
        biased=scores.clone();biased[0,3]+=5
        promoted=sampling_filters({'top_k':1,'top_p':1.0})(ids,biased)
        self.assertEqual(torch.isfinite(promoted).tolist(),[[False,False,False,True]])

    def test_supplement_protocol_and_boundary_interval(self):
        from common import load_protocol
        from analysis import wilson_interval
        nucleus=load_protocol(ROOT/'protocols/nucleus-128.json')
        longer=load_protocol(ROOT/'protocols/nucleus-256.json')
        topk=load_protocol(ROOT/'protocols/topk-128.json')
        self.assertEqual(nucleus['generation']['top_p'],.9)
        self.assertEqual(topk['generation']['top_k'],50)
        self.assertEqual(nucleus['dataset'],longer['dataset'])
        differing={k for k in nucleus['generation'] if nucleus['generation'][k]!=longer['generation'][k]}
        self.assertEqual(differing,{'max_new_tokens'})
        self.assertEqual(nucleus['watermark']['deltas'],[0.,1.,2.,4.])
        lo,hi=wilson_interval(500,500)
        self.assertLess(lo,1);self.assertAlmostEqual(hi,1)

    def test_calibration_and_auc(self):
        z=np.linspace(-3,3,200);threshold=calibrated_threshold(z,.01)
        self.assertLessEqual(float(np.mean(z>threshold)),.01)
        self.assertEqual(auc([2,3],[0,1]),1)
        self.assertEqual(auc([0,1],[0,1]),.5)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
