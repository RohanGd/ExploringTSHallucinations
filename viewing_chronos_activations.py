import h5py

f = h5py.File("chronos_activations.h5", "r")
k = list(f.keys())
print(sorted(k, key = lambda x: (len(x),x)))

import numpy as np


for step in f.keys():
    print(f"\n{step}")
    for layer in f[step].keys():
        dset = f[step][layer]
        print(f"  {layer}: {dset.shape}")
        print(type(dset[:]), dset[:].shape)
    break  # only first step