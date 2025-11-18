import numpy as np



class generator:
    #n = length of copy
    #g = length of time-lag
    #d = delimiter = when should the model output the copy
    #seed = to replicate the same copy
    
    def __init__(self, n, g, d = 9, seed = None):
        self.n = n
        self.g = g
        self.d = d

        np.random.seed(seed)


    def seq(self):
        '''
        input and output sequences are generated for training purposes
        {n sequence to be memorized} + {g time-lag} + {delimiter to activate copying}+{n sequence copy}
        '''
        
        #input sequence
        in_seq = []
        for i in range(self.n):
            in_seq.append(np.random.choice(list('123456789')))
        copy = in_seq.copy()

        in_seq += [0]*self.g

        in_seq.append(self.d)

        in_seq += [0]*self.n


        #output sequence
        out_seq = [0] * (self.n+self.g+1)
        out_seq += copy
        

        

        return in_seq, out_seq

   
        



        
