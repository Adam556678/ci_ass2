import numpy as np

chromosome = [4,1,2]

ranks = np.array(chromosome).argsort().argsort()+1

print(ranks[1])