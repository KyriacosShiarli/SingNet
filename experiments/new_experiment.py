import sys
import os
from os.path import pardir
print pardir
import pdb
from shutil import copy

def make_experiment(name):
  current = os.path.dirname(os.path.abspath(__file__))
  new = current +"/"+name+"/"
  if not os.path.exists(new):
    os.makedirs(new)
    os.makedirs(new+"/plots/")
    os.makedirs(new+"/sound/")
    os.makedirs(new+"/plots/weights")
    os.makedirs(new+"/models/")

    basic_vae_dir = current+"/convVAE.py"
    copy(basic_vae_dir,new)
  else:
    print "Directory already exists, no action taken"

if __name__ == "__main__":
  make_experiment(sys.argv[1])

