from concurrent.futures import ProcessPoolExecutor
from .prediction import evidence, dev_unit, collect
if __name__=='__main__':
    q,_=evidence()
    with ProcessPoolExecutor(max_workers=9) as pool:
        list(pool.map(dev_unit,q['E1']['included']))
    collect()
