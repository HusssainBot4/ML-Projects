import pandas as pd

cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

def load_cmapss(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    return df
