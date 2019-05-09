REDUCE DIMENTIONALITY OF "LARGE" CATEGORICAL FEATURES BY CODING ALL LEVELS WITH N < K ROWS AS "OTHER".

Goal here is to reduce the number of coefficients we need to estimate for categorical fixed effects, 
to allow the model to run quickly on a desktop Python installation. This step is essentially "supervised" 
regularization to reduce number of regressors, and can be eliminated if we're running a model in PySpark with more memory.

Andrew Chamberlain, Ph.D.
May 2019
achamberlain.com
