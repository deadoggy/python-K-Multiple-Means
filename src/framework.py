import numpy as np 
import json 
import sys 
from pyproj import Proj, transform
from catetreedist import CateTree, convertor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import silhouette_score
from weighted_kmm import WeightedKMultipleMeans
import math
import datetime 
import warnings
warnings.filterwarnings("ignore")
sys.path.append('../')

# original data file name
staypoints_fn = 'data/lasvegas_business_info.json'
user_fn = 'data/lasvegas_ub.json'
# preprocessed file name
user_locations_fn = 'data/user_locations.json'
user_addresses_fn = 'data/user_addresses.json'
SIGMA_CATETREE = 1e-3
PREPROCESS = False

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('[%s] %s'%(ts, msg))

#   Step.0 prepare data
def preprocess_data():
    ##  Step.0.1 load data from files  
    log('Step.0.1 Load Data')
    with open(staypoints_fn) as sp_in:
        staypoints = json.load(sp_in)
    with open(user_fn) as user_in:
        user_staypointsid = json.load(user_in)
    ##  Step.0.2 preprocess, convert lon/lat to wsj3857
    log('Step.0.2 Convert to 3857')
    def lonlat_to_3857(lon, lat):
        p1 = Proj(init='epsg:4326')
        p2 = Proj(init='epsg:3857')
        x1, y1 = p1(lon, lat)
        x2, y2 = transform(p1, p2, x1, y1, radians=False)
        return [x2, y2]
    uids = []
    user_locations = []
    user_addresses = []
    location_cache = {}
    for idx, spid in enumerate(staypoints):
        print('loc: %d/%d'%(idx, len(staypoints)))
        location_cache[spid] = lonlat_to_3857(staypoints[spid]['longitude'], staypoints[spid]['latitude'])
    for idx, uid in enumerate(user_staypointsid):
        print('user: %d/%d'%(idx, len(user_staypointsid)))
        uids.append(uid)
        locations = []
        addresses = []
        for spid in user_staypointsid[uid]:
            try:
                locations.append(location_cache[spid])
                addresses.append(staypoints[spid]['address'] )
            except Exception as e:
                print('%s not in business ids'%spid)
        if 0==len(locations) or 0==len(addresses):
            continue
        user_locations.append(locations)
        user_addresses.append(addresses)
    with open(user_locations_fn, 'w') as locout:
        json.dump(user_locations, locout)
    with open(user_addresses_fn, 'w') as addout:
        json.dump(user_addresses, addout)
    return user_locations, user_addresses


if PREPROCESS:
    user_locations, user_addresses = preprocess_data()
else:
    with open(user_locations_fn) as locin:
        user_locations = json.load(locin)
    with open(user_addresses_fn) as addin:
        user_addresses = json.load(addin)

user_locations = np.array(user_locations)
user_addresses = np.array(user_addresses)


#   Step.1 Convert address to vectors using CateTree
log('Step.1 Convert address to vectors using CateTree')
pivots = {} # key:root labels of CateTree, val: CateTree
for address_basket in user_addresses:
    for a in address_basket:
        if a[0] not in pivots:
            pivots[a[0]] = CateTree()
        pivots[a[0]].insert(a[0:-1])

pivots = list(pivots.values())
user_vecs = np.array([convertor(addr, {'pivots': pivots, 'sigma': SIGMA_CATETREE}) for addr in user_addresses])

#   Step.2 clustering user_vecs to find similar pattern
log('Step.2 Clustering user_vecs to find similar pattern')
K = 5
km = KMeans(n_clusters=K)
labels = km.fit_predict(user_vecs)

patterns = []
#   Step.3 For each patterns, employ weighted kmm clustering
log('Step.3 For each patterns, employ weighted kmm clustering')
for pattern_idx in range(K):
    log('Step.3 itr:%d'%pattern_idx)
    indices = np.where(labels==pattern_idx)
    flat_locations = np.concatenate(user_locations[indices], axis=0)
    minmax = MinMaxScaler(feature_range=(0,1000))
    flat_locations = minmax.fit_transform(flat_locations)
    locations, counts = np.unique(flat_locations, axis=0, return_counts=True)
    W = counts/np.sum(counts)
    A_ls = []
    S_ls = []
    lb_ls = []
    proto_lb_ls = []
    sc_ls = []
    # Step.3.1 Run Weighted K Multiple Means
    log('  Step.3.1 Weight KMM')
    for k in range(2, math.floor(np.sqrt(locations.shape[0]))):
        log('   Step.3.1 k:%d'%k)
        n_proto = math.floor(np.sqrt(k*locations.shape[0]))
        nn_k = math.ceil(user_vecs.shape[0]/locations.shape[0])
        wkmm = WeightedKMultipleMeans(k, n_proto=n_proto, nn_k=nn_k, l='auto')
        loc_labels, proto_labels, S, A = wkmm.fit(locations, W)
        log('   Step.3.1 Wkmm done')
        lb_ls.append(loc_labels.tolist())
        proto_lb_ls.append(proto_labels.tolist())
        A_ls.append(A.tolist())
        S_ls.append(S.tolist())
        sc_ls.append(silhouette_score(locations, loc_labels))

    patterns.append({
        'user_idx': indices.tolist(),
        'locations': locations.tolist,
        'weight': W.tolist(),
        'A':A_ls,
        'S':S_ls,
        'lb':lb_ls,
        'proto_lb':proto_lb_ls,
        'sc': sc_ls
    })

with open('result.json', 'w') as out:
    rlt = {
        'user_addresses':user_addresses,
        'user_locations': user_locations,
        'user_vecs':user_vecs,
        'pattern': patterns
    }
    json.dump(rlt, out)





