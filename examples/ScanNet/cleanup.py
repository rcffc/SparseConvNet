import os
import glob
import multiprocessing as mp

mask = '../data/*/*_inst_nostuff.pth'
files = glob.glob(mask)


def f(fn):
    os.remove(fn)

p = mp.Pool(processes=mp.cpu_count())
p.map(f,files)
p.close()
p.join()