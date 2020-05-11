import sys
sys.path.append(sys.path[0] + '/../src/')
from kmm import KMultipleMeans
import numpy as np
import math

# load test dataset

testdataset_fn = 'jain.txt'
data = []
label = []
with open('test/jain.txt') as fin:
    lines = fin.readlines()
    for l in lines:
        l = l.strip().split('\t')
        data.append([eval(v) for v in l[0:2]])
        label.append(eval(l[-1]))
data = np.array(data)

# run clustering

kmm = KMultipleMeans(k=2,n_proto=math.floor(np.sqrt(2*373)), l='auto', nn_k=3)
data_label, proto_label, S, A = kmm.fit(data)

print('s')

