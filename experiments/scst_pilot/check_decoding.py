import torch
from experiment import beams,rollout
class Toy:
    def decode(self,mem,tokens):
        logits=torch.full((len(tokens),len(tokens[0]),5),-10.)
        logits[:,-1,2]=5.;logits[:,-1,3]=4.
        return logits
m=Toy();c={'bos_id':1,'eos_id':2,'pad_id':0,'max_len':5};mem=torch.zeros(1,1,1)
b=beams(m,mem,c,2)
assert b[0][0]==[1,2]
assert all(2 not in seq[1:-1] for seq,_ in b)
g,lp=rollout(m,mem,c);assert g==[1,2] and torch.isfinite(lp)
print('EOS termination, beam score ordering and greedy checks passed')
