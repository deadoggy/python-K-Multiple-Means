import sys
sys.path.append(sys.path[0] + '/../')
import numpy as np

class CateTreeNode:

    def __init__(self, label, parent):
        '''
            init function of CateTreeNode

            @laebl: label of the node
            @parent: parent node
        '''

        self.label = label
        self.parent = parent
        self.bus_cnt = 0 #count of business whose tail category is represented by this node
        self.chd_set = [] #children set
    
    def find_label_in_chd(self, label):
        '''
            find if there exists a node in chd_set whose label is same with @label

            @label: label to find

            #return: None if not exists, else the instance
        '''
        if type(label)!=str:
            raise Exception('label must be str or unicode')
        
        for n in self.chd_set:
            if n.label == label:
                return n
        return None
    
class CateTree:

    def __init__(self):
        '''
            init function of CateTree
        '''
        self.root = CateTreeNode(label='', parent=None)
    
    def insert(self, cate_path):
        '''
            insert a category path to category tree, if the path has been
            inserted, then add 1 to bus_cnt
            
            @cate_path: list, a category path 
        '''

        current_node = self.root
        for i in range(len(cate_path)):
            tmp_label = cate_path[i]
            tmp_chd_set = current_node.chd_set
            
            next_node = None
            for n in tmp_chd_set:
                if n.label == tmp_label:
                    next_node = n
                    break
            
            if next_node is None:
                next_node = CateTreeNode(label=tmp_label, parent=current_node)
                current_node.chd_set.append(next_node)
            current_node = next_node
        current_node.bus_cnt += 1
    
    def similarity(self, path_set):
        '''
            calculate similarity between a user and this tree

            @path_set: list, a set of paths, [ [<path_1>], [<path_2>], ...]

            #return: a float
        '''

        sum_similarity = 0.
        for path in path_set:
            similarity = 1.
            found = True
            p = self.root
            for i in range(len(path)):
                label = path[i]
                user_cate_node = p.find_label_in_chd(label)
                if user_cate_node is not None:
                    bus_share = 0. if user_cate_node.bus_cnt==0 else 1.
                    similarity *= 1./(len(user_cate_node.chd_set) + bus_share)
                    if i == len(path)-1:
                        similarity *= 1.0/user_cate_node.bus_cnt
                    p = user_cate_node
                else:
                    found = False
                    break
            
            if found:
                sum_similarity += similarity
        
        return sum_similarity


def convertor(paths, kwargs):
    '''
        convert a user's category data to data a vector

        @paths: list of category paths
        @kwargs: a dict of other parameter, {'pivots':[], 'sigma': float/list}.

        #return: feature vector
    '''

    pivots = kwargs['pivots']
    try:
        sigma = kwargs['sigma']
    except Exception as e:
        sigma = 0.

    feature_arr = [0.0 for i in range(len(pivots))]
    
    for d in range(len(pivots)):
        t = pivots[d]
        if t.__class__ != CateTree:
            raise Exception('the %d-th pivot not a CateTree' % d)
        feature_arr[d] = t.similarity(paths)
        if sigma != 0.:
            if type(sigma) == list:
                s = sigma[d]  #different sigma value for each dimension 
            else:
                s = sigma
            feature_arr[d] = np.exp(-np.power(feature_arr[d], 2)/(2.0*np.power(s, 2)))
    return list(feature_arr)


    