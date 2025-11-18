import numpy as np


class generator:
    #n_data = amount of sequences
    #n = length of copy
    #g = length of time-lag
    #d = delimiter = when should the model output the copy
    #seed = to replicate the same copy
    
    def __init__(self, n, g, n_data, d = 9, seed = None):
        self.n = n
        self.g = g
        self.d = d
        self.n_data = n_data

        np.random.seed(seed)


    def seq(self):
        '''
        input and output sequences are generated for training purposes
        {n sequence to be memorized} + {g time-lag} + {delimiter to activate copying}+{n sequence copy}
        '''
        
      
        seq = np.random.randint(1, 9, size=(self.n_data, self.n))
        zero1 = np.zeros((self.n_data, self.g-1))
        zero2 = np.zeros((self.n_data, self.g))
        delim = self.d * np.ones((self.n_data, 1))
        zero3 = np.zeros((self.n_data, self.n))
        

        in_seq = np.concatenate((seq, zero1, delim, zero3), axis=1).reshape(self.n_data, -1)
        out_seq = np.concatenate((zero3, zero2, seq),axis = 1).reshape(self.n_data, -1)
        

        return in_seq, out_seq


   
        



        
