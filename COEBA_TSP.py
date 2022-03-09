import random
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import *


def fitness_function(w, s, path):
    total = w[s][path[0]]
    for i in range(len(path)):
        current = path[i]
        if i == len(path) - 1:
            next = s
        else:
            next = path[i+1]
        total += w[current][next]
    
    return total


class Bat:
    def __init__(self, size, v, freq):
        self.size = size
        self.path = []
        self.v = v
        self.freq = freq
        self.fitness = 0
    
    
    def init(self):
        self.path = [i for i in range(self.size)]
        random.shuffle(self.path)
        
        
    def update_fitness(self, w, s, func):
        self.fitness = func(self.path, w, s)
        
        
class Population:
    def __init__(self, size, D):
        self.size = size
        self.bats = [Bat(D, 0,0) for i in range(size)]
        
    def init_pop(self):
        for i in range(self.size):
            self.bats[i].init()


class BatAlgorithm():
    def __init__(self, pop_size, N_Gen, A, r, w):
        self.D = 51
        self.pop_size = pop_size  #population size 
        self.N_Gen = N_Gen  #generations
        self.A = A  #loudness
        self.r0 = r  #pulse rate
        self.r = 0
        self.s = 0
        
        self.w = w

        self.f_min = 0.0  #minimum fitness

        self.pop = Population(self.pop_size, self.D)
        self.best = Bat(51,0,0)


    def best_bat(self):
        j = 0
        for i in range(self.pop_size):
            if self.pop.bats[i].fitness < self.pop.bats[j].fitness:
                j = i
        self.best = self.pop.bats[j]
        
    
    def init_bat(self):
        for bat in self.pop.bats:
            bat.path  = [i for i in range(1, self.D)]
            random.shuffle(bat.path)
            
            bat.fitness = fitness_function(self.w, self.s, bat.path)
        self.best_bat()


    def move_bat(self):
        track_best = []
        self.init_bat()

        for t in range(self.N_Gen):
            print('generation: ' + str(t))
            for i, bat in enumerate(self.pop.bats):
                tmp_x = bat.path.copy()
                distance_x = hamming_distance(self.best.path, tmp_x)
                if distance_x == 0:
                    continue
                
                v = np.random.randint(hamming_distance(self.best.path, tmp_x))
                if v < self.D / 2:
                    for i in range(v):
                        tmp_x = insert_function(tmp_x)
                else:
                    for i in range(v):
                        tmp_x = exchange_funtion(tmp_x)
                
                rand = np.random.uniform()
                if rand > self.r:
                    # local search
                    for i in range(self.D // 10):
                        tmp_x = insert_function(tmp_x)
                
                fitness = fitness_function(self.w, self.s, tmp_x)
                rand = np.random.uniform()
                if rand < self.A and fitness < self.best.fitness:
                    bat.path = tmp_x
                    anpha = 0.98
                    gamma = 0.98
                    self.A *= anpha
                    self.r = self.r0 * (1-math.exp(-gamma*(t/self.N_Gen)))
                    
                    self.best.path = tmp_x
                    self.best.fitness = fitness
            track_best.append(self.best.fitness)
        plt.plot([i for i in range(len(track_best))], track_best)
        plt.show()

        print(self.best.fitness)
        print(self.best.path)
        
        
        
########################################
########################################
########################################
def read_file(file_path):
    file = open(file_path, 'r')
    size = int(file.readline())
    nodes = []
    for i in range(size):
        index, x, y = [int(x) for x in file.readline().split(' ')]
        nodes.append(np.array([x, y]))
        
    best_value = int(file.readline())
    return nodes, best_value
    
    
if __name__ == '__main__':
    file_path = 'E:\HUST\Project3\data\TSP_51.txt'
    nodes, best_value = read_file(file_path)
    number_node = len(nodes)
    w = [[distance(nodes[i], nodes[j]) for j in range(number_node)] for i in range(number_node)]
    start = 0
    
    algorithm = BatAlgorithm(pop_size=200, 
                             N_Gen=100, 
                             A = 0.5, 
                             r = 0.5, 
                             w = w
                            )
    algorithm.move_bat()