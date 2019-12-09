import numpy as np



class KMultipleMeans:

    def __init__(self, proto_sz, nn_k, lamb, sa_tol, sf_tol, 
                    metric=lambda x,y:np.sqrt(np.sum((x-y)**2))):
        '''
            init function of KMultipleMeans

            argv:
                @proto_sz: 
                    int, size of prototypes
                @nn_k:
                    int, number of nearest neighbors
                @lamb: 
                    float, parameter of Laplasian matrix of S
                @sa_tol: 
                    float, tolerance to stop the S-protptype loop
                @sf_tol: 
                    float, tolerance to stop the S-F loop
                @metric: 
                    callable, function to calculate distance btw data points
        '''

        self.proto_sz = proto_sz
        self.nn_k = nn_k
        self.lamb = lamb
        self.sa_tol = sa_tol
        self.sf_tol = sf_tol
        self.metric = metric
        self.labels = None
        self.X = None

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

    def _affinity_matrix(self, prototypes):
        '''
            calculate S matrix

            argv:
                @prototypes: 
                    np.ndarray, shape=(self.proto_sz, N_features)

            return:
                np.ndarray, shape=(self.X.shape[0], self.proto_sz)
        '''

        S = np.zeros((self.X.shape[0], self.proto_sz))

        for i in range(0, S.shape[0]):
            x = self.X[i]

            # calcualte distance btw this point to all prototypes and sort these dists
            # from small to large
            sorted_dists = np.sort(np.array([ self.metric(x, v) for v in prototypes]))

            #calculate S_i
            d_ka1 = sorted_dists[self.nn_k]
            deno = self.nn_k * d_ka1 - np.sum(sorted_dists[0:self.nn_k])
            S[i] = np.array([
                (d_ka1 - sorted_dists[i])/deno if i<self.nn_k else 0
                for i in range(self.proto_sz) 
            ])
        
        return S




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
        prototypes = self._sample_prototypes()

        #step 2: loop for updating S and prototypes
        sa_loss = np.inf
        while True:
            #TODO:
            #step 2.1 update 
            pass

