from joblib import Parallel, delayed
import multiprocessing

def RunInParallel(func, params, numCores = None):
    if not numCores:
        numCores = multiprocessing.cpu_count()	 
    results = Parallel(n_jobs=numCores)(delayed(func)(*param) for param in params)
    return results

