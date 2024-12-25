from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)

foo.lhs_sample(20,continue_run=False,continue_id=10)

foo.analyse('sample')
