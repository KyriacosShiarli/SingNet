import sys
import os
from os.path import pardir
print pardir
import pdb
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.split(current)[0]
print parent
pdb.set_trace()

core = parent+"/core/"
helpers = parent+"/helpers/"
print core
sys.path.append(core)
sys.path.append(helpers)
import updates

