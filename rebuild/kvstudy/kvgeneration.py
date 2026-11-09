"""Cache-aware generation; identical watermark/sampling rules to shared study.
No edits or monkeypatches to the shared generation module are made.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import gc
import time
import torch
from common import ROOT,LOG,read_json,atomic_json,stage_status
from generation import document_seed,row_entropy,sampling_filters
from watermark import Greenlist,clean_content,scored_positions,score
from kvcache import BartCache


@torch.inference_mode()
def decode_cache_batch(model,tokenizer,samples,config,seed,delta):
    g=config['generation'];w=config['watermark'];case=config['cache_condition']
    cross_ratio=config['cache']['cross_ratios'][case];self_ratio=config['cache']['self_ratios'][case]
    device=model.device;b=len(samples);filters=sampling_filters(g)
    gl=Greenlist(model.config.vocab_size,w['gamma'],w['hash_key'])
    special=set(tokenizer.all_special_ids);eos=tokenizer.eos_token_id;bos=tokenizer.bos_token_id
    pad=tokenizer.pad_token_id;start=model.config.decoder_start_token_id
    inputs=tokenizer.pad({'input_ids':[s['source_input_ids'] for s in samples]},padding=True,return_tensors='pt').to(device)
    prev=torch.full((b,1),start,device=device,dtype=torch.long)
    raw=[[start] for _ in samples];trace=[[] for _ in samples];counts=[0]*b;done=[False]*b;last=[None]*b
    rngs=[torch.Generator(device=device).manual_seed(document_seed(s['id'],seed)) for s in samples]
    begin=time.monotonic()
    with BartCache(model,inputs,cross_ratio,self_ratio) as cache:
        for step in range(g['max_new_tokens']):
            logits,cache_step=cache.forward(prev,step)
            if not torch.isfinite(logits).all():raise FloatingPointError('Non-finite decoder logits')
            raw_entropy=row_entropy(logits.softmax(-1));scores=logits/g['temperature']
            for token in special-{eos}:
                if token<scores.shape[-1]:scores[:,token]=-float('inf')
            for i in range(b):
                if counts[i]<g['min_content_tokens']:scores[i,eos]=-float('inf')
            forced=step==0
            if forced:scores.fill_(-float('inf'));scores[:,bos]=0
            base=scores.softmax(-1);masks=gl.batch(prev[:,-1],device)
            probs=filters(prev,scores+delta*masks).softmax(-1)
            if not torch.isfinite(probs).all():raise FloatingPointError('Non-finite sampling probability')
            nxt=[pad if done[i] else int(torch.multinomial(probs[i],1,generator=rngs[i]).item()) for i in range(b)]
            values=torch.stack([raw_entropy,row_entropy(base),(base*masks).sum(-1),base.max(-1).values],1).cpu().tolist()
            for i,token in enumerate(nxt):
                if done[i]:continue
                raw[i].append(token);last[i]=cache_step
                trace[i].append({'raw_index':len(raw[i])-1,'token_id':token,'base_entropy':values[i][0],
                                 'constrained_entropy':values[i][1],'base_green_mass':values[i][2],
                                 'base_p_max':values[i][3],'forced':forced,'special':token in special})
                if token==eos:done[i]=True
                elif token not in special:counts[i]+=1
            prev=torch.tensor(nxt,device=device).unsqueeze(1)
            if all(done):break
        audit=cache.metadata()
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-begin
    wrong=Greenlist(model.config.vocab_size,w['gamma'],w['wrong_key']);rows=[]
    for i,s in enumerate(samples):
        content,positions=clean_content(raw[i],special,eos,start)
        eligible=scored_positions(raw[i],special,eos,start)
        text=tokenizer.decode(content,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        text_ids=tokenizer(text,add_special_tokens=False)['input_ids'];textpos=scored_positions(text_ids,special,eos,start)
        rows.append({'id':s['id'],'role':s['role'],'seed':seed,'delta':delta,'cache_condition':case,
                     'sample_rng_seed':document_seed(s['id'],seed),'summary':text,'raw_ids':raw[i],
                     'content_ids':content,'content_raw_positions':positions,'trace':trace[i],
                     'n_content_tokens':len(content),'ended_with_eos':done[i],'hit_token_limit':not done[i],
                     'watermark':score(raw[i],gl,eligible),'watermark_all_bigrams':score(raw[i],gl,eligible,unique=False),
                     'wrong_key':score(raw[i],wrong,eligible),'text_token_ids':text_ids,
                     'retokenization_changed':text_ids!=content,'watermark_text':score(text_ids,gl,textpos),
                     'wrong_key_text':score(text_ids,wrong,textpos),'batch_size_actual':b,'batch_seconds':elapsed,
                     'cache_at_sequence_end':last[i],'cross_retained_source_positions':audit['cross_retained_source_positions'][i],
                     'source_valid_tokens':audit['cross_valid_source_lengths'][i]})
    return rows,{'sample_ids':[s['id'] for s in samples],**audit}


def safe_decode(model,tokenizer,samples,config,seed,delta):
    try:
        rows,audit=decode_cache_batch(model,tokenizer,samples,config,seed,delta)
        return rows,[audit]
    except torch.cuda.OutOfMemoryError:
        if len(samples)==1:raise
        LOG.warning('KV OOM split microbatch %d',len(samples))
    gc.collect();torch.cuda.empty_cache();k=len(samples)//2
    a,aa=safe_decode(model,tokenizer,samples[:k],config,seed,delta)
    b,bb=safe_decode(model,tokenizer,samples[k:],config,seed,delta)
    return a+b,aa+bb


def generate_condition(model,tokenizer,config,samples,out,identity,smoke=False):
    tasks=[];g=config['generation'];seeds=g['seeds'][:1] if smoke else g['seeds']
    for seed in seeds:
        for delta in config['watermark']['deltas']:
            for start in range(0,len(samples),g['batch_size']):
                p=out/'generation'/f'seed{seed}_delta{delta:g}_batch{start:05}.json'
                batch=samples[start:start+g['batch_size']]
                if p.exists():
                    checkpoint=read_json(p)
                    assert checkpoint['identity']==identity and checkpoint['sample_ids']==[s['id'] for s in batch]
                else:tasks.append((seed,delta,batch,p))
    total=len(seeds)*len(config['watermark']['deltas'])*((len(samples)+g['batch_size']-1)//g['batch_size'])
    already=total-len(tasks);began=time.monotonic()
    LOG.info('KV_GENERATION condition=%s committed=%d pending=%d',config['cache_condition'],already,len(tasks))
    for idx,(seed,delta,batch,path) in enumerate(tasks):
        LOG.info('BATCH_START KV condition=%s seed=%d delta=%g docs=%d',config['cache_condition'],seed,delta,len(batch))
        torch.cuda.reset_peak_memory_stats()
        rows,audits=safe_decode(model,tokenizer,batch,config,seed,delta)
        atomic_json(path,{'identity':identity,'sample_ids':[s['id'] for s in batch],'records':rows,'cache_audits':audits,
                          'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        elapsed=time.monotonic()-began;eta=elapsed/(idx+1)*(len(tasks)-idx-1)
        stage_status(out,'generation',completed_batches=already+idx+1,total_batches=total,eta_seconds=round(eta),cache_condition=config['cache_condition'])
        audit=audits[0];tail=audit['cache_by_decoder_step'][-1]
        LOG.info('PROGRESS KV condition=%s %d/%d ETA=%.0fs cross_width=%d/%d self_kept=%d/%d cache_bytes=%d/%d peak=%.2fGiB',
                 config['cache_condition'],already+idx+1,total,eta,audit['cross_retained_padded_width'],audit['cross_original_padded_width'],
                 tail['self_cache_after_eviction'],tail['positions_seen'],tail['KV_bytes_batch'],tail['full_KV_bytes_batch_at_same_step'],
                 torch.cuda.max_memory_allocated()/2**30)
