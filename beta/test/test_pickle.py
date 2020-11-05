
import os

d = {'mean': [1,2 ,3, 4, 5, 6, 7], 'std': [1, 2, 3, 4, 5, 6, 7]}
path_dir = os.path.abspath(os.path.dirname(__file__))
print (path_dir)
pkl_dir = path_dir + '/test.pkl'
print (pkl_dir)
import pickle
with open(pkl_dir, 'wb') as f:
    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)