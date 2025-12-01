import numpy as np


class generator:
    """
    Generator for the memory copy task.

    Parameters:
        n_data : number of sequences to generate
        n : length of the sequence to memorize
        g : delay length (number of time steps to wait before recall)
        d : delimeter value that triggers recall
        seed : optional seed for reproducibility
    """

    def __init__(self, n, g, n_data, d=9, seed=None):
        self.n = n
        self.g = g
        self.d = d
        self.n_data = n_data
        np.random.seed(seed)

    def seq(self):
        '''
        Generates input and output sequences for training.

        Input format:
            [memory] + [delay zeros] + [delimeter] + [post-delimeter zeros]
        Output format:
            [zeros] + [zeros during delay] + [copied memory]
        
        Returns:
            in_seq : shape (n_data, total_length)
            out_seq : shape (n_data, total_length)
        '''
        
        # Random memory sequences (integers 1-8)
        seq = np.random.randint(1, 9, size=(self.n_data, self.n))

        # Delay and padding zeros
        zero1 = np.zeros((self.n_data, self.g-1))       # delay
        zero2 = np.zeros((self.n_data, self.g))         # delay in output    
        zero3 = np.zeros((self.n_data, self.n))         # post-delimeter zeros

        # Delimeter that signals recall
        delim = self.d * np.ones((self.n_data, 1)) 

        # Construct input: [ memory | delay | delimeter | post-delimeter ]
        in_seq = np.concatenate((seq, zero1, delim, zero3), axis=1).reshape(self.n_data, -1)

        # Construct output: [ zeros | zeros during delay | memory ]
        out_seq = np.concatenate((zero3, zero2, seq),axis = 1).reshape(self.n_data, -1)
        
        return in_seq, out_seq


   
        



        
