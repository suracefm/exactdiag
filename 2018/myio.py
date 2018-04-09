import re
import numpy as np

# Read commented lines of the format '# NAME value' from file .out and return a structured array
def dt(lenarr):
    return np.dtype([('JZZ', np.float64), ('hZ', np.float64), ('hX', np.float64),
               ('FILENAME', np.unicode_, 32), ('phi', np.float64), ('dim_loc', np.int32),
               ('n_dis', np.int32), ('L', np.int32), ('time_set', np.float64, (lenarr,)),
               ('ReZ', np.float64, (lenarr,)), ('ImZ', np.float64, (lenarr,)), ('ReVarZ', np.float64, (lenarr,)),
               ('ImVarZ', np.float64, (lenarr,)), ('ReY', np.float64, (lenarr,)), ('ImY', np.float64, (lenarr,)), ('ReVarY', np.float64, (lenarr,)),
               ('ImVarY', np.float64, (lenarr,)), ('gap', np.float64, (2,)), ('shifted_gap', np.float64, (2,)),
               ('shifted_gap_2', np.float64, (2,)), ('log10_gap', np.float64, (2,)), ('log10_shifted_gap', np.float64, (2,)),
               ('log10_shifted_gap_2', np.float64, (2,)), ('r', np.float64, (2,)), ('lambd', np.float64)])


def get_data(file, dt):
    d = np.zeros([], dtype=dt)
    d['FILENAME']=file
    with open(file) as f:
        for line in f:
            try:
                m = re.match("# (\S+) ([-+]?\d+.\d+e[+-]?\d+)", line)
                d[m.group(1)]=float(m.group(2))
            except:
                try:
                    m = re.match("# (\S+) ([-+]?\d+.\d+)", line)
                    d[m.group(1)]=float(m.group(2))
                except:
                    try:
                        m = re.match("# (\S+) (\d+)", line)
                        d[m.group(1)]=int(m.group(2))
                    except: pass
    d['time_set'], d['ReZ'], d['ImZ'], d['ReVarZ'], d['ImVarZ'],\
    d['ReY'], d['ImY'], d['ReVarY'], d['ImVarY']=np.genfromtxt(file, unpack=True, dtype=np.complex128)
    return d
