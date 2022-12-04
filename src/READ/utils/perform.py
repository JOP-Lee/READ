import os, sys
import time
from collections import defaultdict

import numpy as np


class TicToc:
    def __init__(self):
        self.tic_toc_tic = None

    def tic(self):
        self.tic_toc_tic = time.time()

    def toc(self):
        assert self.tic_toc_tic, 'You forgot to call tic()'
        return (time.time() - self.tic_toc_tic) * 1000

    def tocp(self, str):
        print(f"{str} took {self.toc():.4f}ms")

    @staticmethod
    def print_timing(timing, name=''):
        print(f'\n=== {name} Timimg ===')
        for fn, times in timing.items():
            min, max, mean, p95 = np.min(times), np.max(times), np.mean(times), np.percentile(times, 95)
            print(f'{fn}:\tmin: {min:.4f}\tmax: {max:.4f}\tmean: {mean:.4f}ms\tp95: {p95:.4f}ms')


class AccumDict:
    def __init__(self, num_f=3):
        self.d = defaultdict(list)
        self.num_f = num_f
        
    def add(self, k, v):
        self.d[k] += [v]
        
    def __dict__(self):
        return self.d

    def __getitem__(self, key):
        return self.d[key]
    
    def __str__(self):
        s = ''
        for k in self.d:
            if not self.d[k]:
                continue
            cur = self.d[k][-1]
            avg = np.mean(self.d[k])
            format_str = '{:.%df}' % self.num_f
            cur_str = format_str.format(cur)
            avg_str = format_str.format(avg)
            s += f'{k} {cur_str} ({avg_str})\t\t'
        return s
    
    def __repr__(self):
        return self.__str__()


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'a', buffering=1)
        self.terminal = sys.stdout

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.terminal.write(data)

    def flush(self):
        self.file.flush()
