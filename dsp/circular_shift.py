import numpy as np
from queue import Queue
def main():


    rect = [1,2,3,4,5]

    ## now lets do circular convolution

    permutation_matrix = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
    ])


    rect_mod_shifted = np.dot(permutation_matrix,rect)

    print(rect_mod_shifted)









if __name__ == "__main__":
    main()