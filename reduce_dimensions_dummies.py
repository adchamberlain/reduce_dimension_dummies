# REDUCE DIMENTIONALITY OF "LARGE" CATEGORICAL FEATURES BY CODING ALL LEVELS WITH N < K ROWS AS "OTHER".
# Goal here is to reduce the number of coefficients we need to estimate for categorical fixed effects, 
# to allow the model to run quickly on a desktop Python installation. This step is essentially "supervised" 
# regularization to reduce number of regressors, and can be eliminated if we're running a model in PySpark with more memory.
#
# Andrew Chamberlain, Ph.D.
# May 2019
# achamberlain.com

import pandas as pd
import numpy as np

# Set seed. 
np.random.seed(1)

# Create example dataframe, 100 categories, 1000 observations. 
df = pd.DataFrame(np.random.randint(1, 101, 1000), columns=['level'])
df['level'] = df['level'].astype('category')

# Set K sample size threshold for estimating fixed effects (need at least K observations to be included as a dummy).
K = 10

# Features with N >= K observations
levels_n = pd.DataFrame(df.groupby(['level']).size().reset_index() )
levels_n.columns = ['level','count']
levels_n = levels_n.sort_values('count', ascending = False)
levels_n = levels_n[levels_n['count'] >= K ]

# Check how many levels will be estimated after trimming.
print("Levels with sufficient sample sizes: ", levels_n.shape[0])

# Recode levels with insufficient sample size as "other".
levels_tup = tuple(levels_n['level'])

def level_recode(x):
   if x in levels_tup:
       return x
   else: 
       return 'other'

df['level_recode'] = df['level'].apply(lambda x: level_recode(x) )

# Check number of levels, including 'other', to be estimated. 
print("Original number of levels: ", len(set(df['level'])))
print("Levels to be estimated after trimming: ", len(set(df['level_recode'])))