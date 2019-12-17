import sys
sys.path.append(sys.path[0] + '/../src/')
from kmm import KMultipleMeans
import numpy as np

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

kmm = KMultipleMeans(proto_sz=13, nn_k=3)
data_label, proto_label, S = kmm.fit(data, 2)

print('s')

