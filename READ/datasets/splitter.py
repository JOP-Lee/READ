import numpy as np
import os



def split_by_ratio(lists, train_ratio):
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    
    splits = []
    train_inds, val_inds = [], []

    train_n = int(sz[0] * train_ratio)
    train_inds, val_inds = np.split(np.random.permutation(sz[0]), [train_n])

    print(train_inds)
    print( val_inds)

    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[val_inds]])
        
    return splits



#100frame test
def split_by_step100(lists, val_step, train_drop):
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    
    splits = []
    train_inds, val_inds = [], []
    #print('lists',lists)
    for i in range(sz[0]):
        if 5<(i % 100) < 95 or i<10:
            train_inds.append(i)
        else:
            val_inds.append(i)

    print(train_inds)
    print( val_inds)

    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[val_inds]])
        
    return splits


#正常
def split_by_step(lists, val_step, train_drop):
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'
    
    splits = []
    train_inds, val_inds = [], []

    for i in range(sz[0]):
        if i % val_step == 0:
            val_inds.append(i)
        elif train_drop < i % val_step < val_step - train_drop:
            train_inds.append(i)

    print(train_inds)
    print( val_inds)

    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[val_inds]])
        
    return splits
