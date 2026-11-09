"""Batched cached seq2seq sampling with explicit decoding controls and token traces.
One independent RNG per document/seed, reset for each delta. No KV compression.
"""
import gc
import time
import torch
from transformers import (AutoModelForSeq2SeqLM,GenerationConfig,LogitsProcessor,
                          LogitsProcessorList,TopKLogitsWarper,TopPLogitsWarper)
from common import ROOT, LOG, atomic_json, read_json, digest, stage_status
from watermark import Greenlist, clean_content, scored_positions, score


def document_seed(sample_id, seed):
    return int(digest({'id':sample_id,'seed':seed})[:15],16)


def row_entropy(p):
    return -(p*p.clamp_min(1e-30).log()).sum(-1)


def sampling_filters(config):
    """HF order: custom watermark bias, then top-k, then top-p, then sampling."""
    top_k=int(config['top_k']);top_p=float(config['top_p'])
    if top_k<0 or not 0<top_p<=1:raise ValueError('Invalid sampling filter')
    filters=LogitsProcessorList()
    if top_k>0:filters.append(TopKLogitsWarper(top_k=top_k,min_tokens_to_keep=1))
    if top_p<1:filters.append(TopPLogitsWarper(top_p=top_p,min_tokens_to_keep=1))
    return filters


@torch.inference_mode()
def decode_batch(model, tokenizer, samples, config, seed, delta):
    g=config['generation']; w=config['watermark']; device=model.device
    filters=sampling_filters(g)
    gl=Greenlist(model.config.vocab_size,w['gamma'],w['hash_key'])
    specials=set(tokenizer.all_special_ids)
    eos,pad,bos,start=tokenizer.eos_token_id,tokenizer.pad_token_id,tokenizer.bos_token_id,model.config.decoder_start_token_id
    if None in (eos,pad,bos,start):
        raise ValueError('Missing BART special token config')
    inputs=tokenizer.pad({'input_ids':[s['source_input_ids'] for s in samples]},padding=True,return_tensors='pt').to(device)
    encoder=model.get_encoder()(**inputs,return_dict=True)
    b=len(samples); prev=torch.full((b,1),start,device=device,dtype=torch.long)
    raw=[[start] for _ in samples]; trace=[[] for _ in samples]; content_counts=[0]*b
    generators=[torch.Generator(device=device).manual_seed(document_seed(s['id'],seed)) for s in samples]
    done=[False]*b; past=None
    t0=time.monotonic()
    for step in range(g['max_new_tokens']):
        output=model(encoder_outputs=encoder,attention_mask=inputs.attention_mask,
                     decoder_input_ids=prev,past_key_values=past,use_cache=True,return_dict=True)
        past=output.past_key_values
        logits=output.logits[:,-1,:].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite model logits')
        p_raw=logits.softmax(-1)
        entropy_raw=row_entropy(p_raw)
        scores=logits/g['temperature']
        for tok in specials-{eos}:
            if tok < scores.shape[-1]:
                scores[:,tok]=-float('inf')
        for i in range(b):
            if content_counts[i]<g['min_content_tokens']:
                scores[i,eos]=-float('inf')
        # BART standard forced BOS; its deterministic position is never tested.
        forced=(step==0)
        if forced:
            scores.fill_(-float('inf'));scores[:,bos]=0
        base=scores.softmax(-1)
        masks=gl.batch(prev[:,-1],device)
        q=(base*masks).sum(-1)
        # Filter AFTER adding watermark bias, exactly as HF generate's warper order.
        # q/base entropy above describe the pre-filter distribution, not final sampling.
        sampling_scores=filters(prev,scores+delta*masks)
        probs=sampling_scores.softmax(-1)
        if not torch.isfinite(probs).all():
            raise FloatingPointError('Non-finite sampling probabilities')
        next_ids=[]
        for i in range(b):
            tok=pad if done[i] else int(torch.multinomial(probs[i],1,generator=generators[i]).item())
            next_ids.append(tok)
        small=torch.stack([entropy_raw,row_entropy(base),q,base.max(-1).values],dim=1).cpu().tolist()
        for i,tok in enumerate(next_ids):
            if done[i]:
                continue
            raw[i].append(tok)
            trace[i].append({'raw_index':len(raw[i])-1,'token_id':tok,
                             'base_entropy':small[i][0], 'constrained_entropy':small[i][1],
                             'base_green_mass':small[i][2], 'base_p_max':small[i][3],
                             'forced':forced,'special':tok in specials})
            if tok==eos:
                done[i]=True
            elif tok not in specials:
                content_counts[i]+=1
        prev=torch.tensor(next_ids,device=device).unsqueeze(1)
        if all(done):
            break
    torch.cuda.synchronize()
    elapsed=time.monotonic()-t0
    records=[]
    wrong=Greenlist(model.config.vocab_size,w['gamma'],w['wrong_key'])
    for i,s in enumerate(samples):
        content,positions=clean_content(raw[i],specials,eos,start)
        eligible=scored_positions(raw[i],specials,eos,start)
        detected=score(raw[i],gl,eligible)
        summary=tokenizer.decode(content,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        # A deployable detector sees displayed text, not private sampled token IDs.
        text_ids=tokenizer(summary,add_special_tokens=False)['input_ids']
        text_positions=scored_positions(text_ids,specials,eos,start)
        records.append({'id':s['id'],'role':s['role'],'seed':seed,'delta':delta,
                        'sample_rng_seed':document_seed(s['id'],seed),'summary':summary,
                        'raw_ids':raw[i],'content_ids':content,'content_raw_positions':positions,
                        'trace':trace[i],'n_content_tokens':len(content),'ended_with_eos':done[i],
                        'hit_token_limit':not done[i],'watermark':detected,
                        'text_token_ids':text_ids,'retokenization_changed':text_ids!=content,
                        'watermark_text':score(text_ids,gl,text_positions),
                        'wrong_key_text':score(text_ids,wrong,text_positions),
                        'watermark_all_bigrams':score(raw[i],gl,eligible,unique=False),
                        'wrong_key':score(raw[i],wrong,eligible),
                        'batch_size_actual':b,'batch_seconds':elapsed})
    return records


def decode_with_oom(model, tokenizer, samples, config, seed, delta):
    try:
        return decode_batch(model,tokenizer,samples,config,seed,delta)
    except torch.cuda.OutOfMemoryError:
        if len(samples)==1:
            raise
        LOG.warning('CUDA OOM: split microbatch %d; per-document RNG restarts deterministically',len(samples))
    # Leave the except block before retrying, so its traceback releases GPU tensors.
    gc.collect();torch.cuda.empty_cache()
    k=len(samples)//2
    return (decode_with_oom(model,tokenizer,samples[:k],config,seed,delta)+
            decode_with_oom(model,tokenizer,samples[k:],config,seed,delta))


@torch.inference_mode()
def validate_native_decoder(model, tokenizer, samples, config, delta=0.0):
    """Real-checkpoint engineering test, separate from experimental measurements."""
    from copy import deepcopy
    cfg=deepcopy(config);cfg['generation']['max_new_tokens']=32
    sample=samples[:1]
    ours=decode_batch(model,tokenizer,sample,cfg,42,delta)[0]
    eos=tokenizer.eos_token_id;bos=tokenizer.bos_token_id
    special=set(tokenizer.all_special_ids)
    minimum=cfg['generation']['min_content_tokens']
    reference_green=Greenlist(model.config.vocab_size,cfg['watermark']['gamma'],cfg['watermark']['hash_key'])
    class MatchConstraints(LogitsProcessor):
        def __call__(self,input_ids,scores):
            for token in special-{eos}:
                if token<scores.shape[-1]:scores[:,token]=-float('inf')
            lexical=torch.ones_like(input_ids,dtype=torch.bool)
            for token in special:lexical &= input_ids.ne(token)
            scores[lexical.sum(-1)<minimum,eos]=-float('inf')
            if input_ids.shape[1]==1:
                scores.fill_(-float('inf'));scores[:,bos]=0
            return scores+delta*reference_green.batch(input_ids[:,-1],scores.device)
    inputs=tokenizer.pad({'input_ids':[s['source_input_ids'] for s in sample]},padding=True,return_tensors='pt').to(model.device)
    generation_config=GenerationConfig(do_sample=True,num_beams=1,temperature=cfg['generation']['temperature'],
                                      top_k=cfg['generation']['top_k'],top_p=cfg['generation']['top_p'],
                                      max_new_tokens=32,pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,
                                      bos_token_id=bos,decoder_start_token_id=model.config.decoder_start_token_id,
                                      use_cache=True)
    torch.manual_seed(document_seed(sample[0]['id'],42))
    native=model.generate(**inputs,generation_config=generation_config,logits_processor=LogitsProcessorList([MatchConstraints()]))[0].tolist()
    if ours['raw_ids']!=native:
        raise AssertionError('Custom sampler differs from HF generate on matched singleton configuration')
    encoder=model.get_encoder()(**inputs,return_dict=True);past=None;maximum=0.0
    for step in range(min(16,len(native)-1)):
        prefix=torch.tensor([native[:step+1]],device=model.device)
        cached=model(encoder_outputs=encoder,attention_mask=inputs.attention_mask,decoder_input_ids=prefix[:,-1:],past_key_values=past,use_cache=True)
        full=model(encoder_outputs=encoder,attention_mask=inputs.attention_mask,decoder_input_ids=prefix,use_cache=False)
        err=float((cached.logits[:,-1,:]-full.logits[:,-1,:]).abs().max())
        maximum=max(maximum,err);past=cached.past_key_values
    if maximum>0.125:raise AssertionError(f'Full-prefix/cache FP16 logit discrepancy: {maximum}')
    return {'sample_id':sample[0]['id'],'delta':delta,
            'top_k':cfg['generation']['top_k'],'top_p':cfg['generation']['top_p'],
            'sampled_ids_match_HF_generate':True,'max_cache_full_logit_abs_error':maximum,
            'comparison_fp16_absolute_tolerance':0.125,'tested_decoder_steps':min(16,len(native)-1),
            'purpose':'engineering_only_not_scientific_evidence'}


def generate_all(config, samples, tokenizer, out, identity, smoke=False):
    tasks=[]; g=config['generation']
    seeds=g['seeds'][:1] if smoke else g['seeds']
    for seed in seeds:
        for delta in config['watermark']['deltas']:
            for start in range(0,len(samples),g['batch_size']):
                batch=samples[start:start+g['batch_size']]
                path=out/'generation'/f'seed{seed}_delta{delta:g}_batch{start:05}.json'
                tasks.append((seed,delta,batch,path))
    missing=[]
    for task in tasks:
        if task[3].exists():
            data=read_json(task[3])
            assert data['identity']==identity and data['sample_ids']==[s['id'] for s in task[2]], 'Checkpoint identity error'
        else:
            missing.append(task)
    LOG.info('GENERATION checkpointed=%d pending=%d total=%d',len(tasks)-len(missing),len(missing),len(tasks))
    if not missing:
        return
    model=AutoModelForSeq2SeqLM.from_pretrained(ROOT/config['model']['directory'],local_files_only=True,
                                             torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    # bart-large-cnn's tokenizer includes <mask> (50264), while its trained
    # decoder has 50264 logits. Do not resize trained embeddings to accommodate it.
    assert len(tokenizer)>=model.config.vocab_size
    for sample in samples:
        if max(sample['source_input_ids'])>=model.config.vocab_size:
            raise ValueError(f'Source contains an unsupported added token: {sample["id"]}')
    atomic_json(out/'model_config.json',model.config.to_dict())
    if smoke:
        checks=[validate_native_decoder(model,tokenizer,samples,config,delta) for delta in (0.0,2.0)]
        atomic_json(out/'native_decoder_check.json',{'checks':checks})
        LOG.info('NATIVE_DECODER_CHECK %s',checks)
    start_time=time.monotonic(); finished=len(tasks)-len(missing)
    for idx,(seed,delta,batch,path) in enumerate(missing):
        LOG.info('BATCH_START phase=generation seed=%d delta=%g docs=%d checkpoint=%s',seed,delta,len(batch),path.name)
        torch.cuda.reset_peak_memory_stats()
        rows=decode_with_oom(model,tokenizer,batch,config,seed,delta)
        atomic_json(path,{'identity':identity,'sample_ids':[s['id'] for s in batch],'records':rows,
                          'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        done=finished+idx+1; duration=time.monotonic()-start_time
        eta=(len(missing)-idx-1)*duration/(idx+1)
        stage_status(out,'generation',completed_batches=done,total_batches=len(tasks),eta_seconds=round(eta))
        LOG.info('PROGRESS generation %d/%d batches %.1f%% elapsed=%.1fs ETA=%.1fs peak=%.2fGiB',
                 done,len(tasks),100*done/len(tasks),duration,eta,torch.cuda.max_memory_allocated()/2**30)
    del model;gc.collect();torch.cuda.empty_cache()
