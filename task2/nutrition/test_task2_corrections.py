import json
import unittest
from pathlib import Path
import numpy as np
from model_caption import build_vocab_from_train, EOS
from metrics import ingredient_f1
from predict import select_b_ingredients

class Corrections(unittest.TestCase):
    def test_schema_grammar_and_projection(self):
        import random
        from constrained_caption import Grammar,reconcile_weights
        vocab=build_vocab_from_train()
        names=json.loads((Path(__file__).parent/'ingredients.json').read_text())
        for seed in range(100):
            rng=random.Random(seed);g=Grammar(vocab,names);tokens=[]
            for _ in range(2048):
                if not g.q:break
                t=rng.choice(g.allowed());g.advance(t);tokens.append(t)
            else:self.fail('Grammar did not terminate')
            obj=json.loads(vocab.decode(tokens));actual=[x['name'] for x in obj['items']]
            self.assertEqual(len(actual),len(set(actual)))
            self.assertTrue(set(actual)<=set(names))
            out=reconcile_weights(obj)
            self.assertAlmostEqual(sum(x['weight_g'] for x in out['items']),out['total_weight_g'],places=6)
        out=reconcile_weights({'items':[{'name':'a','weight_g':0},{'name':'b','weight_g':0}],'total_weight_g':.3})
        self.assertEqual([x['weight_g'] for x in out['items']],[.2,.1])

    def test_training_json_roundtrip(self):
        vocab=build_vocab_from_train()
        for line in (Path(__file__).parent/'annotations/train.jsonl').read_text(encoding='utf-8').splitlines():
            text=json.loads(line)['target_text']
            self.assertEqual(json.loads(vocab.decode(vocab.encode(text))),json.loads(text))
        ids=vocab.encode('{"name":"bell peppers"}')
        self.assertEqual(vocab.decode(ids+[EOS]+ids),vocab.decode(ids))

    def test_macro_includes_missed_and_spurious_classes(self):
        m=ingredient_f1([['a','c'],[]],[['a'],['b']], labels=['a','b','c'])
        self.assertAlmostEqual(m['macro_f1'],1/3)
        self.assertEqual(m['macro_class_count'],3)
        self.assertAlmostEqual(m['micro_f1'],.5)

    def test_cap_uses_confidence(self):
        p=np.linspace(.5,.99,20)
        self.assertEqual(set(select_b_ingredients(p,.4)),set(range(4,20)))
        self.assertEqual(len(select_b_ingredients(p,1)),0)
        self.assertEqual(list(select_b_ingredients(np.ones(20),.5)),list(range(16)))

if __name__=='__main__':unittest.main()
