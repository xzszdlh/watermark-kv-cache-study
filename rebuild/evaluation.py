"""ROUGE vs reference + sentence source-support NLI proxy.
All visible source tokens are covered by overlapping NLI windows; nothing silently
truncates. NLI is a model-based support proxy, not human factuality ground truth.
"""
import gc
import math
import time
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from nltk.tokenize.punkt import PunktSentenceTokenizer,PunktParameters
from rouge_score.rouge_scorer import RougeScorer
from common import ROOT,LOG,read_json,atomic_json,stage_status
from watermark import Greenlist,scored_positions,score
from alignment import original_token_offsets,sentence_assignment


def splitter():
    params=PunktParameters()
    params.abbrev_types={'mr','mrs','ms','dr','prof','inc','ltd','jr','sr','st','u.s','u.k','vs'}
    return PunktSentenceTokenizer(params)


def sentence_units(row, source, bart_tokenizer, gl, config):
    text=row['summary']; enc=bart_tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets,invalid_utf8=original_token_offsets(bart_tokenizer,row['content_ids'],text)
    raw=row['raw_ids']; rawmap=row['content_raw_positions']
    allpos=scored_positions(raw,set(bart_tokenizer.all_special_ids),bart_tokenizer.eos_token_id,2)
    text_ids=enc['input_ids']
    text_valid=set(scored_positions(text_ids,set(bart_tokenizer.all_special_ids),bart_tokenizer.eos_token_id,2))
    valid=set(allpos); trace={v['raw_index']:v for v in row['trace']}
    source3=set(zip(source['source_input_ids'],source['source_input_ids'][1:],source['source_input_ids'][2:]))
    units=[]
    spans=list(splitter().span_tokenize(text))
    assigned=sentence_assignment(offsets,spans)
    text_assigned=sentence_assignment(enc['offset_mapping'],spans)
    for sentence_index,(start,end) in enumerate(spans):
        ci=assigned[sentence_index]
        positions=[rawmap[i] for i in ci if rawmap[i] in valid]
        det=score(raw,gl,positions)
        detall=score(raw,gl,positions,unique=False)
        toks=[row['content_ids'][i] for i in ci]
        triples=list(zip(toks,toks[1:],toks[2:]))
        unit={'sentence_index':sentence_index,'char_start':start,'char_end':end,'text':text[start:end],
              'relative_position':sentence_index/max(len(spans)-1,1),
              'alignment_valid':True,'alignment_method':'original_BPE_bytes_to_decoded_characters',
              'canonical_retokenization_matches':text_ids==row['content_ids'],
              'summary_invalid_utf8_sequences':invalid_utf8,
              'watermark':det,
              'watermark_text':score(text_ids,gl,[i for i in text_assigned[sentence_index] if i in text_valid]),
              'mean_base_entropy':sum(trace[rawmap[i]]['base_entropy'] for i in ci)/len(ci) if ci else None,
              'copy_trigram_fraction':sum(x in source3 for x in triples)/len(triples) if triples else 0.0,
              'repeat_fraction':1-det['n']/detall['n'] if detall['n'] else 0.0,
              'n_content_tokens':len(ci)}
        units.append(unit)
    return units


class NLIEvaluator:
    def __init__(self,config):
        self.config=config
        path=ROOT/config['nli']['directory']
        self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,
                                                                    torch_dtype=torch.float16).to('cuda').eval()
        labels={int(i):v.lower() for i,v in self.model.config.id2label.items()}
        LOG.info('NLI labels=%s',labels)
        # Fail rather than guess if the checkpoint uses ambiguous LABEL_0 etc.
        self.entail=next((i for i,v in labels.items() if v=='entailment'),None)
        self.contradict=next((i for i,v in labels.items() if v=='contradiction'),None)
        if self.entail is None or self.contradict is None:
            raise ValueError(f'Unverified NLI label mapping: {labels}')
        self.cache={}

    def windows(self,source):
        if source['id'] not in self.cache:
            # This is tokenization only. The full source is split below before any
            # model forward; keep the explicit <=512 assertion on actual pairs.
            ids=self.tok(source['visible_source'],add_special_tokens=False,verbose=False)['input_ids']
            e=self.config['evaluation']; windows=[]
            for start in range(0,len(ids),e['premise_stride']):
                part=ids[start:start+e['premise_chunk_tokens']]
                if not part:break
                windows.append((start,part))
                if start+len(part)==len(ids):break
            if not windows:raise ValueError('Empty model-visible source')
            assert windows[0][0]==0 and windows[-1][0]+len(windows[-1][1])==len(ids)
            assert all(b[0]<=a[0]+len(a[1]) for a,b in zip(windows,windows[1:])), 'Source coverage gap'
            self.cache[source['id']]=windows
        return self.cache[source['id']]

    @torch.inference_mode()
    def predict(self,features):
        if not features:return []
        try:
            batch=self.tok.pad(features,padding=True,return_tensors='pt').to('cuda')
            assert batch.input_ids.shape[1]<=512
            probs=self.model(**batch).logits.float().softmax(-1)
            if not torch.isfinite(probs).all():raise FloatingPointError('Non-finite NLI probability')
            return probs.cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            if len(features)==1:raise
            LOG.warning('NLI OOM reduce microbatch %d',len(features))
        # Release exception frames before allocating the smaller microbatch.
        gc.collect();torch.cuda.empty_cache()
        mid=len(features)//2
        return self.predict(features[:mid])+self.predict(features[mid:])

    def evaluate_units(self,records,sources):
        features=[]; owners=[]; metadata=[]
        e=self.config['evaluation']
        for ri,r in enumerate(records):
            windows=self.windows(sources[r['id']])
            for ui,u in enumerate(r['sentences']):
                u['nli']=None
                hyp=self.tok(u['text'],add_special_tokens=False)['input_ids']
                if not hyp or len(hyp)>e['max_hypothesis_tokens']:
                    u['nli_missing_reason']='empty_or_hypothesis_over_limit'
                    continue
                for wi,(offset,premise) in enumerate(windows):
                    f=self.tok.prepare_for_model(premise,pair_ids=hyp,add_special_tokens=True,
                                                 truncation=False,return_attention_mask=True)
                    if len(f['input_ids'])>512:raise ValueError('NLI pair exceeds declared maximum')
                    features.append(dict(f)); owners.append((ri,ui)); metadata.append((wi,offset,len(premise)))
        grouped={}
        bs=e['nli_batch_size']
        for start in range(0,len(features),bs):
            preds=self.predict(features[start:start+bs])
            for owner,meta,p in zip(owners[start:start+bs],metadata[start:start+bs],preds):
                grouped.setdefault(owner,[]).append((meta,p))
        for (ri,ui),entries in grouped.items():
            best=max(entries,key=lambda v:v[1][self.entail]); meta,p=best
            evidence=self.windows(sources[records[ri]['id']])[meta[0]][1]
            records[ri]['sentences'][ui]['nli']={
                'source_support_proxy':p[self.entail],
                'contradiction_at_best_window':p[self.contradict],
                'windows_evaluated':len(entries),
                'best_window_index':meta[0],'best_window_token_offset':meta[1],
                'best_evidence_text':self.tok.decode(evidence,skip_special_tokens=True),
                'all_window_probabilities':[{'window':m[0],'entailment':v[self.entail],
                                             'contradiction':v[self.contradict]} for m,v in entries]}
        return len(features)

    def sanity(self):
        premise='Alice bought three red apples. Bob lives in Paris.'
        hypotheses=['Alice bought three red apples.','Alice bought seven blue apples.']
        features=[dict(self.tok(premise,h,return_attention_mask=True,truncation=False)) for h in hypotheses]
        p=self.predict(features)
        result={'premise':premise,'hypotheses':hypotheses,'probabilities':p,
                'entailment_index':self.entail,'contradiction_index':self.contradict}
        if not (p[0][self.entail]>p[1][self.entail] and p[1][self.contradict]>p[0][self.contradict]):
            raise RuntimeError(f'NLI direction sanity failed: {result}')
        return result


def evaluate_all(config,samples,bart_tokenizer,out,identity):
    sources={s['id']:s for s in samples}; tasks=[]
    for p in sorted((out/'generation').glob('*.json')):
        dest=out/'evaluation'/p.name
        if dest.exists():
            assert read_json(dest)['identity']==identity
        else:tasks.append((p,dest))
    LOG.info('EVALUATION pending batches=%d',len(tasks))
    if not tasks:return
    evaluator=NLIEvaluator(config);atomic_json(out/'nli_sanity.json',evaluator.sanity())
    decoder_vocab=read_json(ROOT/config['model']['directory']/'config.json')['vocab_size']
    gl=Greenlist(decoder_vocab,config['watermark']['gamma'],config['watermark']['hash_key'])
    scorer=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    started=time.monotonic()
    for idx,(path,dest) in enumerate(tasks):
        raw=read_json(path);assert raw['identity']==identity
        records=[]
        for r in raw['records']:
            source=sources[r['id']]
            rouge={k:v.fmeasure for k,v in scorer.score(source['reference'],r['summary']).items()}
            units=sentence_units(r,source,bart_tokenizer,gl,config)
            records.append({'id':r['id'],'role':r['role'],'seed':r['seed'],'delta':r['delta'],
                            'rouge':rouge,'sentences':units})
        LOG.info('BATCH_START phase=NLI checkpoint=%s docs=%d',path.name,len(records))
        pairs=evaluator.evaluate_units(records,sources)
        atomic_json(dest,{'identity':identity,'records':records,'nli_pairs':pairs})
        elapsed=time.monotonic()-started;eta=elapsed/(idx+1)*(len(tasks)-idx-1)
        stage_status(out,'evaluation',completed_pending_batches=idx+1,total_pending_batches=len(tasks),eta_seconds=round(eta))
        LOG.info('PROGRESS evaluation %d/%d batches pairs=%d elapsed=%.1fs ETA=%.1fs',idx+1,len(tasks),pairs,elapsed,eta)
    del evaluator;gc.collect();torch.cuda.empty_cache()
