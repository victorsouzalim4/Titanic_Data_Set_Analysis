import numpy as np

def compare_classifications(results1, results2):
    
    r1 = np.array(results1)
    r2 = np.array(results2)

    if r1.shape != r2.shape:
        raise ValueError("the vectors must have the same lenght")

    equals = np.sum(r1 == r2)
    return equals

