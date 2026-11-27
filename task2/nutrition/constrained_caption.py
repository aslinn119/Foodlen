"""Schema-constrained token decoding for the existing CNN/GRU, no labels at inference."""
from collections import deque
from decimal import Decimal, ROUND_HALF_UP
import json
import re
import copy
import torch
from metrics import tokenize, NUM_KEYS

class Grammar:
    def __init__(self,vocab,names,max_items=16):
        self.vocab=vocab;self.names={tuple(vocab.word2idx[t] for t in tokenize(n)):n for n in names}
        if len(self.names)!=len(names):raise ValueError('Ambiguous ingredient tokenization')
        self.used=set();self.prefix=();self.max_items=max_items
        self.q=deque();self.add('{"items":[{"name":"');self.q.append('NAME');self.add('","weight_g":');self.q.extend(['NUM','AFTER_ITEM'])
        self.digits=[i for i,w in enumerate(vocab.idx2word) if re.fullmatch(r'0|[1-9][0-9]*',w)]
        self.fraction=[i for i,w in enumerate(vocab.idx2word) if w.isdigit()]
        self.dot=vocab.word2idx['.'];self.decimal=False
    def add(self,text):self.q.extend(self.vocab.word2idx[t] for t in tokenize(text))
    def allowed(self):
        s=self.q[0]
        if isinstance(s,int):return [s]
        if s=='NAME':
            return sorted({k[len(self.prefix)] if len(k)>len(self.prefix) else self.vocab.word2idx['"'] for k,n in self.names.items() if n not in self.used and k[:len(self.prefix)]==self.prefix})
        if s=='NUM':return self.digits
        if s=='FRAC':return self.fraction
        if s=='NUM_END':return [self.dot,self._next_delimiter()]
        if s=='AFTER_ITEM':return [self.vocab.word2idx['}']]
        if s=='NEXT_ITEM':return [self.vocab.word2idx[']']] + ([self.vocab.word2idx[',']] if len(self.used)<min(self.max_items,len(self.names)) else [])
        raise RuntimeError(s)
    def _next_delimiter(self):return self.vocab.word2idx['}'] if self.q[1]=='AFTER_ITEM' else self.q[1]
    def advance(self,token):
        s=self.q[0]
        if isinstance(s,int):self.q.popleft();return
        if s=='NAME':
            if token==self.vocab.word2idx['"']:
                self.used.add(self.names[self.prefix]);self.prefix=();self.q.popleft();self.q.popleft()
            else:self.prefix+= (token,)
        elif s=='NUM':self.q[0]='NUM_END'
        elif s=='NUM_END':
            if token==self.dot:self.q[0]='FRAC'
            else:self.q.popleft();self.advance(token)
        elif s=='FRAC':self.q.popleft()
        elif s=='AFTER_ITEM':self.q[0]='NEXT_ITEM'
        elif s=='NEXT_ITEM':
            self.q.popleft()
            if token==self.vocab.word2idx[',']:
                self.add('{"name":"');self.q.append('NAME');self.add('","weight_g":');self.q.extend(['NUM','AFTER_ITEM'])
            else:
                for k in NUM_KEYS:self.add(',"'+k+'":');self.q.append('NUM')
                self.add('}');self.q.append(2)

@torch.no_grad()
def generate_constrained(model,images,vocab,names):
    model.eval();states=[Grammar(vocab,names) for _ in range(len(images))];seqs=[[] for _ in states]
    h=model.encode(images);x=torch.ones((len(states),1),dtype=torch.long,device=images.device)
    for _ in range(2048):
        out,h=model.gru(model.embedding(x),h);logits=model.fc_out(out[:,-1]).detach().cpu()
        nxt=[]
        for i,g in enumerate(states):
            if not g.q:nxt.append(2);continue
            ids=g.allowed();t=ids[int(logits[i,ids].argmax())];g.advance(t);seqs[i].append(t);nxt.append(t)
        if all(not g.q for g in states):return seqs
        x=torch.tensor(nxt,device=images.device).unsqueeze(1)
    raise RuntimeError('Constrained decoder exceeded structural length bound')

def reconcile_weights(obj):
    """Declared output projection: preserve predicted total; exact tenths by largest remainder."""
    result=copy.deepcopy(obj);items=result['items']
    if not items:raise ValueError('Cannot reconcile empty ingredient list without inventing an ingredient')
    total=int((Decimal(str(max(result['total_weight_g'],0)))*10).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
    weights=[max(float(i['weight_g']),0) for i in items];denom=sum(weights)
    shares=[total*w/denom for w in weights] if denom else [total/len(items)]*len(items)
    ints=[int(x) for x in shares]
    for i in sorted(range(len(items)),key=lambda i:-(shares[i]-ints[i]))[:total-sum(ints)]:ints[i]+=1
    for i,n in zip(items,ints):i['weight_g']=n/10
    result['total_weight_g']=total/10
    return result
