import numpy as np
from tarjan import TarjanSCC

class KMultipleMeans:
    '''
        ref: 
            Nie, Feiping & Wang, Cheng-Long & Li, Xuelong. (2019). 
            K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.
            959-967. 10.1145/3292500.3330846. 
    '''

    def __init__(self, k, n_proto, nn_k, r='auto', l=1., 
                    metric=lambda x,y:np.sqrt(np.sum((x-y)**2))):
        '''
            init function of KMultipleMeans

            argv:
                @k:
                    int, number of clusters
                @n_proto: 
                    int, size of prototypes
                @nn_k:
                    int, number of nearest prototypes which are connnected to each data point
                @r: 
                    float/string, the parameter of L2 regularization, if can be a float or 'auto'.
                @l: 
                    float, the parameter of eigen values item
                @metric: 
                    callable, function to calculate distance btw data points,
        '''
        self.k = k
        self.n_proto = n_proto
        self.nn_k = nn_k
        self.r = r
        self.l = l
        self.metric = metric


    def fit(self, X):
        '''
            cluster the data points to k clusters

            argv:
                @X:
                    np.ndarray, shape=(N_datapoints, N_features), data points

            return:
                (data_labels, prototype_labels), 
                    where shape(data_labels)=(N_datapoints,), shape(prototype_labels)=(n_proto,)
        '''

        # Step.1: pick @n_proto samples randomly as the prototypes
        A = X[np.random.choice(X.shape[0], self.n_proto, replace=False),]
        S = np.zeros((X.shape[0], A.shape[0]))
    
        # Step.2 Optimization
        while True:
        
            # Step.2.1 solve the assignment of neighboring prototypes in problem(4)
            sum_deno = 0.
            for i in range(X.shape[0]):
                dist_to_proto = np.array([self.metric(X[i], A[j]) for j in range(A.shape[0])])
                sorted_dist_idx = np.argsort(dist_to_proto)
                denominator = self.n_proto * dist_to_proto[sorted_dist_idx[self.n_proto]]\
                    - np.sum( dist_to_proto[ sorted_dist_idx[0:self.n_proto], ] )
                sum_deno += denominator

                for j in range(self.n_proto):
                    S[i][sorted_dist_idx[j]] = (
                            dist_to_proto[sorted_dist_idx[self.n_proto]] - dist_to_proto[sorted_dist_idx[j]]
                        ) / denominator
            if self.r == 'auto':
                r = sum_deno / (2*X.shape[0])
            else:
                r = self.r
            
            # Step.2.2 Fix A, update S,F
            while True:
                
