# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:47:02 2020

@author: meizihang
"""


import toad
import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
import os
from sklearn import tree  

def DTR_TO_SQL(tree, feature_names, when_else=-1):
    import numpy as np
    
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]   

    le='<='               
    g ='>'
    
    idx = np.argwhere(left == -1)[:,0]     

    def find_value(tree):
        value_lis=[]
        tree_ = tree.tree_
        def recurse(node, depth):
            from sklearn.tree import _tree
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                recurse(tree_.children_left[node], depth + 1)
                recurse(tree_.children_right[node], depth + 1)
            else:
                value_lis.append(tree_.value[node][0][0])
        recurse(0, 1)
        return value_lis

    value_lis = find_value(tree)
    
    
    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = [child]
        
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        
        lineage.append((parent, split, threshold[parent], features[parent]))
        
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    print('case ')
    for j,child in enumerate(idx):
        clause=' when '
        for node in recurse(left, right, child):
            if len(str(node))<3:
                continue
            i=node
            if i[1]=='l':  
                sign=le 
            else: 
                sign=g
            clause=clause+i[3]+sign+str(i[2])+' and '
        clause=clause[:-4]+' then '+ str(value_lis[j])
        print(clause)
    print('else %s end as clusters'%(when_else))
    
