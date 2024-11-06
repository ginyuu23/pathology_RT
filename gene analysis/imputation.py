import sys
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import numpy as np
import pandas as pd
import math
from missingpy import MissForest
import warnings
warnings.simplefilter('ignore', FutureWarning)

df_incom = pd.read_csv("expression_drop80.csv",header=0,index_col=0)

data = df_incom.to_numpy(dtype=float)

print("start processing...")
imputer = MissForest(max_iter=50, decreasing=True,
                     n_estimators=100, max_depth=5,
                     max_features=1, oob_score=True)
full=imputer.fit_transform(data)

np.savetxt('expression_missf0907.csv',full,delimiter=',')

print("finished")
