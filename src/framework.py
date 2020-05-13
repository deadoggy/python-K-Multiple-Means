import numpy as np 
import json 
import sys 
from pyproj import Proj, transform
from catetreedist import CateTree, convertor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../')

staypoints_fn = 'data/lasvegas_business_info.json'
user_fn = 'data/lasvegas_ub.json'
SIGMA_CATETREE = 1e-3

#   Step.0 prepare data
##  Step.0.1 load data from files  
with open(staypoints_fn) as sp_in:
    staypoints = json.load(sp_in)
with open(user_fn) as user_in:
    user_staypointsid = json.load(user_in)
##  Step.0.2 preprocess, convert lon/lat to wsj3857
def lonlat_to_3857(lon, lat):
    p1 = Proj(init='epsg:4326')
    p2 = Proj(init='epsg:3857')
    x1, y1 = p1(lon, lat)
    x2, y2 = transform(p1, p2, x1, y1, radians=True)
    return [x2, y2]
uids = []
user_locations = []
user_addresses = []
for uid in user_staypointsid:
    uids.append(uid)
    locations = []
    addresses = []
    for spid in user_staypointsid[uid]:
        if spid in staypoints:
            locations.append({'id': spid, 'loc':lonlat_to_3857(staypoints[spid]['longitude'], staypoints[spid]['latitude'])})
            addresses.append(staypoints[spid]['address'] for spid in user_staypointsid[uid])
    user_locations.append(locations)
    user_addresses.append(addresses)


#   Step.1 Convert address to vectors using CateTree
pivots = {} # key:root labels of CateTree, val: CateTree
for address_basket in user_addresses:
    for a in address_basket:
        if a[0] not in pivots:
            pivots[a[0]] = CateTree()
        pivots[a[0]].insert(a[0:-1])

pivots = list(pivots.values())
user_vecs = np.array([convertor(addr, {'pivots': pivots, 'sigma': SIGMA_CATETREE}) for addr in user_addresses])

#   Step.2 clustering user_vecs to find similar pattern

K = 5
km = KMeans(n_clusters=K)
labels = km.fit_predict(user_vecs)

#   Step.3 For each patterns, employ weighted kmm clustering
for pattern_idx in range(K):
    indices = np.where(labels==pattern_idx)
    pattern_locations = {} # key: id, val: dict: {'loc':[a,b], 'cnt':int}
    for i in range(user_vecs.shape[0]):



