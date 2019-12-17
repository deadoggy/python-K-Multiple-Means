import numpy as np


class TarjanSCC:
    '''
        Tarjan's algorithm to solve the strongest connection components(SCC) problem of a directed graph

        ref:
            https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
            https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components
    '''

    def __init__(self, affinity_threshold=0.):
        '''
            init function of TarjanSCC

            argv:
                @affinity_threshold:
                    float, a threshold. If the affinity btw two nodes is equal to less than this value, 
                    the corresponding edge will be ignored 
        ''' 

        self.affinity_threshold = affinity_threshold
        self.index = 0
    
    def _is_connected(self, affinity):
        '''
            judge if the @affinity is greater than self.affinity_threshold

            argv:
                @affinity:
                    float, the affinity(weight) value btw two nodes
        '''
        return affinity > self.affinity_threshold

    def fit(self, mat):
        '''
            run Tarjan's algorithm

            argv:
                @mat:
                    np.ndarray, shape=(N_datapoints, N_datapoints), 
                    mat[i][j] is the affinity value btw i-th node and j-th node

            return:
                np.ndarray, shape=(N_datapoints,), labels of nodes
        '''

        size = mat.shape[0]
        node_idx = np.array([-1 for i in range(size)])
        node_low = np.array([-1 for i in range(size)])
        node_stk = np.array([False for i in range(size)])
        stack = []

        label = np.array([-1 for i in range(size)])

        def _scc(v):
            node_idx[v] = self.index
            node_low[v] = self.index
            self.index += 1
            stack.append(v)
            node_stk[v] = True

            for w in range(size):
                if not self._is_connected(mat[v][w]):
                    continue
                if node_idx[w]==-1:
                    _scc(w)
                    node_low[v] = min(node_low[v], node_low[w])
                elif node_stk[w]: 
                    #   Node_stk[w] is True means w is in this SCC and it has been processed, so v is 
                    # one of w's descendants now <v,w> exists means <v,w> is a back edge, in SCC case,
                    # low value means topmost reachable ancestor (with minimum possible Disc value) 
                    # via the subtree of that node(include the self node), so the second term should 
                    # be node_idx[w]
                    # 
                    #   Actually, if we only consider SCC problem, the node_idx[w] can be replaced 
                    # by node_low[w]. In this case, the definition of node_low changes and it's more 
                    # conveninent to retrieve results (Nodes in a common SCC share a common low value). 
                    # 
                    #   However, the prior defintion of node_low can be applied into other problems such 
                    # as articulation point, bridge and biconnected component (They can be solved in a 
                    # common [node_idx+node_low] framework), but the second defnition can't.
                    #   
                    #   In this implementation, the first definition is employed.
                    node_low[v] = min(node_low[v], node_idx[w])
            
            if node_idx[v] == node_low[v]:
                scc_idx = []
                while True:
                    w = stack.pop()
                    node_stk[w] = False
                    scc_idx.append(w)
                    if w==v:
                        label[scc_idx] = v
                        break
        
        for v in range(size):
            if node_idx[v] == -1:
                _scc(v)
        
        return label


if __name__=='__main__':

    mat = np.array([[0,0,0, 0],[0,0,1,0],[0,0,0,1],[0,1,0,0]])
    tc = TarjanSCC(0.)
    label = tc.fit(mat)
    print(label.tolist())









            
    
    