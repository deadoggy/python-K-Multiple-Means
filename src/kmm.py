import numpy as np
from tarjan import TarjanSCC

class KMultipleMeans:
    '''
        ref: 
            Nie, Feiping & Wang, Cheng-Long & Li, Xuelong. (2019). 
            K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters.
            959-967. 10.1145/3292500.3330846. 
    '''

    def __init__(self, k, n_proto, nn_k, tol=1e-1, l=1000., max_itr_1=15, max_itr_2=15, 
                    metric=lambda x,y:np.sum((x-y)**2)):
        '''
            init function of KMultipleMeans

            argv:
                @k:
                    int, number of clusters
                @n_proto: 
                    int, size of prototypes
                @nn_k:
                    int, number of nearest prototypes which are connnected to each data point
                @tol: 
                    float, if abs of difference between two data is less than @tol, they are treated as the same.
                @l: 
                    float/'auto', the parameter of eigen values item; if 'auto', then the l will be r first, then
                    in each iteration, if the number of zero-eigvals larger than k, divided by 2; if less than k, 
                    multiplied by 2.
                @max_itr_1:
                    int, the maximum times of iterations of the outer loops
                @max_itr_2:
                    int, the maximum tiems of iterations of the inner loops
                @metric: 
                    callable, function to calculate distance btw data points,
        '''
        self.k = k
        self.n_proto = n_proto
        self.nn_k = nn_k
        self.tol = tol
        self.l = l
        self.max_itr_1 = max_itr_1
        self.max_itr_2 = max_itr_2
        self.metric = metric

    def _solve_Si(self, dist_to_proto):
        '''
            solve the problem(4) and the problem(20) in the paper

            argv:
                @dist_to_proto: np.ndarray, shape=(self.n_proto), the distances between a data point
                                and all prototypes
            
            return:
                (Si, denominator), S is a np.ndarray with shape=(self.n_proto,), denominator is a float 
                which is used to calcualte the r parameter if needed 
        '''
        Si = np.zeros_like(dist_to_proto)
        sorted_dist_idx = np.argsort(dist_to_proto)
        denominator = self.nn_k * dist_to_proto[sorted_dist_idx[self.nn_k]]\
            - np.sum( dist_to_proto[ sorted_dist_idx[0:self.nn_k], ] )

        for j in range(self.nn_k):
            Si[sorted_dist_idx[j]] = (
                    dist_to_proto[sorted_dist_idx[self.nn_k]] - dist_to_proto[sorted_dist_idx[j]]
                ) / denominator

        return Si, denominator

    def fit(self, X):
        '''
            cluster the data points to k clusters

            argv:
                @X:
                    np.ndarray, shape=(N_datapoints, N_features), data points

            return:
                (data_labels, prototype_labels, S, A), 
        '''

        # Step.1: pick @n_proto samples randomly as the prototypes
        A = X[np.random.choice(X.shape[0], self.n_proto, replace=False),]
        S = np.zeros((X.shape[0], A.shape[0]))
    
        # Step.2 Optimization
        itr_1 = 0
        while itr_1<self.max_itr_1:
            
            # Step.2.0 calculate distances between data points and prototypes, which is used in both Step.2.1 and Step.2.2
            dist_to_proto = np.zeros((X.shape[0], self.n_proto))
            for i in range(X.shape[0]):
                dist_to_proto[i] = np.array([self.metric(X[i], A[j]) for j in range(A.shape[0])])
    
            # Step.2.1 solve the assignment of neighboring prototypes in problem(4)
            sum_deno = 0.
            for i in range(X.shape[0]):
                S[i], deno= self._solve_Si(dist_to_proto[i])
                sum_deno += deno
            
            if self.l == 'auto':
                l = sum_deno/(2.*X.shape[0])
            else:
                l = self.l

            # Step.2.2 Fix A, update S,F
            itr_2 = 0
            while itr_2<self.max_itr_2:

                # Step.2.2.1 Fix S, update F
                P = np.zeros((X.shape[0]+self.n_proto, X.shape[0]+self.n_proto))
                P[0:X.shape[0], X.shape[0]:] = S
                P[X.shape[0]:, 0:X.shape[0]] = S.T
                D = np.zeros_like(P)
                for i in range(P.shape[0]):
                    D[i][i] = 1/ np.sqrt(np.sum(P[i,:]))
                D[D==np.inf] = 0.
                Du = D[0:X.shape[0], 0:X.shape[0]]
                Dv = D[X.shape[0]:, X.shape[0]:]
                Sh = Du.dot(S).dot(Dv)
                U,M,VT = np.linalg.svd(Sh) # According to the doc, columns of U and rows of VT are singular vectors
                F = (np.sqrt(2)/2)*np.concatenate((U[:,0:self.k], VT.T[:,-self.k:]), axis=0)

                # Step.2.2.2 Fix F, update S
                for i in range(X.shape[0]):
                    # calculate vij in problem(17)
                    n = X.shape[0]
                    v = np.array([ np.sum((F[i]*D[i][i]-F[n+j]*D[n+j][n+j])**2) for j in range(self.n_proto)])
                    S[i], deno = self._solve_Si(dist_to_proto[i]+l*v)

                # Step.2.2.3 judge whether to stop the iterations
                L = np.identity(X.shape[0]+self.n_proto) - D.dot(P).dot(D)
                sorted_eigvals = np.sort(np.linalg.eigvals(L))
                zero_eig_cnt = len(sorted_eigvals[sorted_eigvals<self.tol])
                itr_2 += 1
                if zero_eig_cnt == self.k:
                    break
                else:
                    if self.l=='auto':
                        if zero_eig_cnt > self.k:
                            l /= 2
                        else:
                            l *= 2

            
            # Step.2.3 Fix S,F, update A
            old_A = A.copy()
            for i in range(self.n_proto):
                A[i] = X.T.dot(S[:,i])/np.sum(S[:,i])

            itr_1 += 1
            # Step.2.4 judge whether to stop the iteration
            if np.sum(np.abs(old_A-A))<self.tol:
                break
            
        
        # Step.3 Use tarjan algorithm to solve the SCC problem
        tarjan = TarjanSCC(self.tol)
        P = np.zeros((X.shape[0]+self.n_proto, X.shape[0]+self.n_proto))
        P[0:X.shape[0], X.shape[0]:] = S
        P[X.shape[0]:, 0:X.shape[0]] = S.T
        all_labels = tarjan.fit(P)

        # save S, A, l
        self.S = S
        self.A = A
        self.l = l

        return (all_labels[0:X.shape[0]], all_labels[X.shape[0]:], S, A)

            




