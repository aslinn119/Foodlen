import unittest,ast,json
from pathlib import Path
from release_tools import align_predictions,score,ROOT
class ReleaseTests(unittest.TestCase):
    def test_alignment(self):
        refs=[{'dish_id':'a'},{'dish_id':'b'}]
        self.assertEqual([r['dish_id'] for r in align_predictions(list(reversed(refs)),refs)],['a','b'])
        for bad in [[refs[0]],refs+[refs[0]],[{'dish_id':'c'},refs[0]]]:
            with self.assertRaises(ValueError):align_predictions(bad,refs)
    def test_fixed_macro(self):
        # Evaluate actual metrics.py function without importing its numpy dependency.
        tree=ast.parse((ROOT/'metrics.py').read_text(encoding='utf-8'))
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ingredient_f1')
        class NP:
            @staticmethod
            def mean(xs):return sum(xs)/len(xs)
        ns={'np':NP,'json':json,'__file__':str(ROOT/'metrics.py')}
        exec(compile(ast.Module(body=[fun],type_ignores=[]),'metrics.py','exec'),ns)
        m=ns['ingredient_f1']([['bacon']],[['bacon']])
        self.assertAlmostEqual(m['macro_f1'],1/151);self.assertEqual(m['micro_f1'],1)
        self.assertEqual(m['macro_class_count'],151)
        with self.assertRaises(ValueError):ns['ingredient_f1']([], [['bacon']])
if __name__=='__main__':unittest.main()
