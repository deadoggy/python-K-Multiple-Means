import numpy as np



class KMultipleMeans:
    '''
        ref: 
            Nie, Feiping & Wang, Cheng-Long & Li, Xuelong. (2019). 
            K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.
            959-967. 10.1145/3292500.3330846. 
    '''

    def __init__(self, proto_sz, nn_k, 
                    metric=lambda x,y,ix,iy:np.sum((x-y)**2)):
        '''
            init function of KMultipleMeans

            argv:
                @proto_sz: 
                    int, size of prototypes
                @nn_k:
                    int, number of nearest neighbors
                @metric: 
                    callable, function to calculate distance btw data points,
                    arguments are: v1, v2, idx_v1, idx_v2
        '''

        self.proto_sz = proto_sz
        self.nn_k = nn_k
        self.metric = metric
        self.labels = None
        self.A = None
        self.X = None
        self.S = None
        self.F = None
        self.r = None

    def _sample_prototypes(self):
        '''
            sample <self.proto_sz> prototypes from self.X
        '''
        prototypes = np.zeros((self.proto_sz, self.X.shape[1]))
        prototypes[0:self.proto_sz] = self.X[0:self.proto_sz]

        for i in range(self.proto_sz, self.X.shape[0]):
            idx = np.random.randint(0, self.proto_sz)
            if idx < self.proto_sz:
                prototypes[idx] = self.X[i]
        
        return prototypes

    def _sparsity_parameter(self, sorted_distance):
        '''
            calculate the sparsity regularization based on a sorted distance list

            argv:
                @sorted_distance:
                    np.ndarray, shape=(self.proto_sz,)
        '''
        item_1 = sorted_distance[self.nn_k]
        item_2 = np.sum(sorted_distance[0:self.nn_k])
        return (self.nn_k * item_1 - item_2) / 2.

    def _sparsity_item(self, metric):
        '''
            calculate matrix S and sparsity parameter

            argv:
                @metric:
                    callable, distance calculating function
            return:
                (S, r)
                S: np.ndarray, shape=(self.X.shape[0], self.proto_sz)
                r: float
        '''

        S = np.zeros((self.X.shape[0], self.proto_sz))
        r = 0.

        for i in range(0, S.shape[0]):
            x = self.X[i]

            # calcualte distance btw this point to all prototypes and sort these dists
            # from small to large
            sorted_dists = np.sort(np.array([ metric(x, v, i, iv) for iv, v in enumerate(self.A)]))

            #calculate S_i
            d_ka1 = sorted_dists[self.nn_k]
            deno = self.nn_k * d_ka1 - np.sum(sorted_dists[0:self.nn_k])
            S[i] = np.array([
                (d_ka1 - sorted_dists[i])/deno if i<self.nn_k else 0
                for i in range(self.proto_sz) 
            ])

            #calculate r_i
            r += self._sparsity_parameter(sorted_dists)
        
        return (S, r/self.X.shape[0])

    def _degree_matrix(self, mat):
        '''
            calculate degree matrix of @mat

            argv:
                @mat: 
                    2D np.ndarray
            
            return:
                np.ndarray, degree matrix of mat
        '''

        degree_mat = np.zeros(mat.shape)
        for i in range(degree_mat.shape[0]):
            degree_mat[i][i] = np.sum(mat[i])
        return degree_mat

    def _square_affinity_matrix(self):
        '''
            calculate P matrix in paper

            return:
                np.ndarray, shape=(N_datapoints+self.proto_sz, N_datapoints+self.protosz)
        '''
        n = self.S.shape[0]
        m = self.S.shape[1]

        P = np.zeros((n+m, n+m))
        P[0:n, n: ] = self.S
        P[n: , 0:n] = self.S.T

        return P

    def _laplacian(self):
        '''
            calculate normalized laplacian matrix of P

            return:
                np.array, shape=(self.X.shape[0]+self.proto_sz, self.X.shape[0]+self.proto_sz)
        '''
        P = self._square_affinity_matrix()
        D = self._degree_matrix(P)
        D_s = D**(-1/2)
        D_s[D_s==np.inf] = 0.

        return np.identity(P.shape[0]) - D_s.dot(P).dot(D_s) 

    def fit(self, X, k):
        '''
            run the KMultipleMeans clustersing algorithm

            argv:
                @X: 
                    np.ndarray, shape=(N_datapoints, N_features)
                @k: 
                    int, size of clusters
            
            return:
                self.labels
        '''

        self.X = X
        self.labels = np.zeros((X.shape[0]))        
        # step 1: Initialize multiple-means
        self.A = self._sample_prototypes()

        #step 2: loop for updating S and prototypes
        lamb = np.inf
        while True:

            #step 2.1 calculate S and r by the optimal solution to problem (4) in paper
            self.S, self.r = self._sparsity_item(self.metric) 
            lamb = self.r

            #step 2.2 fix A, update S & F
            while True:
                # step 2.2.1 fix S, update F
                D = self._degree_matrix(self._square_affinity_matrix())
                Ds = D**(-1/2)
                Ds[Ds==np.inf] = 0.
                Du_s = Ds[0              :self.X.shape[0], 0              :self.X.shape[0]]
                Dv_s = Ds[self.X.shape[0]:               , self.X.shape[0]:               ]
                _S = Du_s.dot(self.S).dot(Dv_s)
                u, sval, vh = np.linalg.svd(_S)
                
                U = (np.sqrt(2)/2.) * u[:, :k] # U.shape=(N, k)
                V = (np.sqrt(2)/2.) * vh[:k, :].T # V.shape=(M, k)
                self.F = np.concatenate((U,V), axis=0)

                # step 2.2.2 fix F, update S
                def updateS_metric(v1, v2, idx_v1, idx_v2):
                    # item 1
                    item_1 = self.metric(v1, v2, idx_v1, idx_v2)

                    # item 2
                    i = idx_v1
                    j = idx_v2 + self.X.shape[0]
                    f_i = self.F[i] / np.sqrt(D[i][i])
                    f_j = self.F[j] / np.sqrt(D[j][j])
                    item_2 = np.sum((f_i-f_j)**2)

                    return item_1 + lamb * item_2
                
                self.S, self.r = self._sparsity_item(updateS_metric)
                
                # check whether to stop the iteration
                L = self._laplacian()
                eigval, eigvec = np.linalg.eig(L)
                zero_eig_cnt = eigval.shape[0]-np.count_nonzero(eigval)
                if k == zero_eig_cnt:
                    break
                elif k>zero_eig_cnt:
                    lamb *= 2
                else:
                    lamb /= 2
            
            #step 2.3 fix S & F, update A
            is_break = True
            for i in range(self.A.shape[0]):
                new_Ai = (self.S[:, i].T.dot(self.X)) / (np.sum(self.S[:,i]))
                if is_break and new_Ai != self.A[i]:
                    is_break = False
                self.A[i] = new_Ai

            if is_break:
                break
        
        # step 3, assign labels
        # 