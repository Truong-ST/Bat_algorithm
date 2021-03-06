import random 
import numpy as np


class Bat:
    def __init__(self, gen, v, freq, func):
        self.gen = gen # [priority_i for i in V]
        # self.path = self.get_path()
        self.path = []
        self.v = v
        self.freq = freq
        self.fitness = 0
        self.func = func
    
    
    def init(self):
        return 
        
        
    def update_fitness(self, w, s):
        self.fitness = self.func(self.path, w, s)
        
        
class Population:
    def __init__(self, size):
        self.size = size
        self.population = [Bat() for i in range(size)]


class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function):
        self.D = D  #dimension
        self.NP = NP  #population size 
        self.N_Gen = N_Gen  #generations
        self.A = A  #loudness
        self.r = r  #pulse rate
        self.Qmin = Qmin  #frequency min
        self.Qmax = Qmax  #frequency max
        self.Lower = Lower  #lower bound
        self.Upper = Upper  #upper bound

        self.f_min = 0.0  #minimum fitness
        
        self.Lb = [0] * self.D  #lower bound
        self.Ub = [0] * self.D  #upper bound
        self.Q = [0] * self.NP  #frequency

        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  #velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.NP)]  #population of solutions
        self.Fitness = [0] * self.NP  #fitness
        self.best = [0] * self.D  #best solution
        self.Fun = function


    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    def init_bat(self):
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = np.random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.D, self.Sol[i])
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def move_bat(self):
                # Base
        # track_best = []
        # for t in range(self.N_Gen):
        #     print('generation: ' + str(t))
        #     for i, bat in enumerate(self.pop):
        #         tmp_x = bat.path.copy()
        #         print(tmp_x)
        #         distance_x = hamming_distance(self.best, tmp_x)
        #         if distance_x == 0:
        #             continue

        #         v = np.random.randint(distance_x)
        #         if v < self.number_node / 2:
        #             for i in range(v):
        #                 tmp_x = insert_function(tmp_x)
        #         else:
        #             for i in range(v):
        #                 tmp_x = exchange_funtion(tmp_x)

        #         rand = np.random.uniform()
        #         if rand  > self.r:
        #             # local search
        #             tmp_x = insert_function(tmp_x)

        #         fitness = self.fitness_path(tmp_x)
        #         rand = np.random.uniform()
        #         if rand < self.A and fitness < self.f_min:
        #             bat.path = tmp_x
        #             anpha = 0.98
        #             gamma = 0.98
        #             self.A *= anpha
        #             self.r = self.r0 * (1-math.exp(-gamma*(t/self.N_Gen)))

        #             self.best.path = tmp_x
        #             self.best.fitness = fitness
        #     track_best.append(self.f_min)
        
        
        
        # tmp solution
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]

        self.init_bat()

        for t in range(self.N_Gen):
            for i in range(self.NP):
                # update infor
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * np.random.uniform(0, 1)
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] - self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]
                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])

                # local search
                rnd = np.random.random_sample()
                if rnd > self.r:
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])
                        
                Fnew = self.Fun(self.D, S[i])

                # check to update solution if better and random
                rnd = np.random.random_sample()
                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew
                    # increase ri, decrease Ai

                # update best
                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

        print(self.f_min)
        
        
        
########################################
########################################
########################################


# Fitness function
def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

# For reproducive results
#random.seed(5)

for i in range(10):
    #  D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function
    Algorithm = BatAlgorithm(D=10, NP=40, N_Gen=1000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, Lower=-10.0, Upper=10.0, function=Fun)
    Algorithm.move_bat()