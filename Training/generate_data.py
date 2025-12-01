import numpy as np
from generator_memory import generate_memory_task

"""
Generate data of T length to train. The data are sequences of characters that first consists
of a short segment of information that have to be stored by the mode, followed by a varying 
length of blank inputs, then a marker value that informs the model when the stored information 
should be recalled.


For example:

Input:
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 A

Output:

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
"""
