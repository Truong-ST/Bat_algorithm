import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from algorithm import *
from copy import deepcopy
import glob
import os
import itertools


anpha = 0.95
gamma = 0.95

class Bat:
    def __init__(self, gen, v, freq):
        self.gen = gen  # [priority_i for i in V]
        self.path = []
        self.v = v
        self.freq = freq
        self.fitness = 0

    def update_fitness(self, w, s):
        self.fitness = self.func(self.path, w, s)


class BatAlgorithm():
    def __init__(self, domains, nodes, w, adjacent_list, s, t, NP, N_Gen, A, r, Qmin, Qmax):
        # process graph
        self.w = w  # weight
        self.nodes = nodes  # node
        self.s = s  # start
        self.t = t  # end
        self.domain = domains  # domain correspond node
        self.number_node = len(nodes)
        self.number_domain = len(self.domain)
        self.node_of_domain = self.node_domain()
        self.link_domain = self.get_link_domain()
        self.adjacent_list = adjacent_list
        
        self.domain_start = self.node_of_domain[s]
        self.domain_end = self.node_of_domain[t]
        self.list_domain = [i for i in range(self.number_domain)]
        self.order_domain = [i for i in range(self.number_domain - 1)]

        ## hyperparameter
        self.NP = NP  # population size
        self.N_Gen = N_Gen  # generations
        self.A = [A for i in range(NP)]  # loudness
        self.r0 = r # pulse rate
        self.r = [0 for i in range(NP)]
        self.Qmin = Qmin  # frequency min
        self.Qmax = Qmax  # frequency max

        self.total_priority_domain = [1 for i in range(self.number_domain)]
        self.pop = [Bat([0], [0], 0) for i in range(self.NP)]

        # best
        self.len_top = int(0.1 * NP)
        self.top_best = []
        self.best = self.pop[0]  # x



    def get_top_best(self):
        self.pop = sorted(self.pop, key=lambda x: x.fitness)[:self.NP]
        self.top_best = self.pop[:self.len_top]
        self.best = self.top_best[0]


    # def best_bat(self):
    #     j = 0
    #     for i in range(self.NP):
    #         if self.pop[i].fitness < self.pop[j].fitness:
    #             j = i
    #     self.best = self.pop[j].path
    #     self.f_min = self.pop[j].fitness


    def node_domain(self):
        """
        defind node is in which domain
        """
        node_of_domain = [0 for i in range(self.number_node)]
        # for i in range(self.number_domain):
            # for v in self.domain[i]:
            #     node_of_domain[v] = i
        for i, d in enumerate(self.domain):
            for v in d:
                node_of_domain[v] = i

        return node_of_domain


    def get_link_domain(self):
        """
        get matrix link of domain
        """
        link_domain = [[0 for i in range(self.number_domain)]
                       for j in range(self.number_domain)]
        for i in range(self.number_node):
            d1 = self.node_of_domain[i]
            for j in range(self.number_node):
                d2 = self.node_of_domain[j]
                if d1 == d2:
                    continue
                link_domain[d1][d2] = 1
        return link_domain


    def convert_gen_domain(self, gen):
        """
        decode gen to path
        """
        start_domain = self.node_of_domain[self.s]

        path = [start_domain]
        not_visited = [i for i in range(self.number_domain)]
        not_visited.remove(start_domain)
        for i in range(self.number_domain-1):
            next_domain = -1
            best_priority = -1
            for j in not_visited:
                if gen[j] > best_priority:
                    best_priority = gen[j]
                    next_domain = j

            path.append(next_domain)
            not_visited.remove(next_domain)
        return path
        

    def gen_subgraph(self, path):
        """
        generate sub graph from path
        """
        weight = [[0 for i in range(self.number_node)]
                  for j in range(self.number_node)]
        for i in range(1, self.number_domain):
            current_d = path[i-1]
            d = path[i]
            list_node_current_d = self.domain[current_d]
            list_node_d = self.domain[d]
            for node in list_node_current_d:
                # neighbor_node = self.adjacent_list[node]
                # next_node = set(neighbor_node) & set(list_node_current_d) | set(list_node_d)
                # for j in next_node:
                #     weight[node][j] = self.w[node][j]
                for j in list_node_d:
                    weight[node][j] = self.w[node][j]
                for j in list_node_current_d:
                    weight[node][j] = self.w[node][j]
                    
            if current_d == self.domain_end:
                break
                
        return weight
    
    
    def gen_subgraph_new(self, path):
        """
        generate sub graph from path
        """
        weight = [[0 for i in range(self.number_node)]
                  for j in range(self.number_node)]
                
        path_new = path.copy()
        path_new.insert(0, self.domain_start)
        for i in range(1, self.number_domain):
            current_d = path_new[i-1]
            d = path_new[i]
            list_node_current_d = self.domain[current_d]
            list_node_d = self.domain[d]
            for node in list_node_current_d:
                # neighbor_node = self.adjacent_list[node]
                # next_node = set(neighbor_node) & set(list_node_current_d) | set(list_node_d)
                # for j in next_node:
                #     weight[node][j] = self.w[node][j]
                for j in list_node_d:
                    weight[node][j] = self.w[node][j]
                for j in list_node_current_d:
                    weight[node][j] = self.w[node][j]
                
            if current_d == self.domain_end:
                break
                
        return weight


    def init_bat(self):
        for bat in self.pop:
            bat.gen = [i for i in range(self.number_domain)]
            random.shuffle(bat.gen)
            
            bat.path_domain = self.convert_gen_domain(bat.gen)
            subgraph = self.gen_subgraph(bat.path_domain)
            
            # bat.gen.remove(self.domain_start)
            # subgraph = self.gen_subgraph_new(bat.gen)
            
            bat.fitness = idpc_dijkstra(subgraph, self.s, self.t)

        self.get_top_best()


    def move_bat(self):
        self.init_bat()

        track_best = []
        for t in range(self.N_Gen):
            # print('generation: ' + str(t))
            for i in range(self.NP):
                bat = deepcopy(self.pop[i])
                
                ### update to near best
                # bat.freq = self.Qmin + (self.Qmax - self.Qmin) * np.random.uniform(0, 1)
                # bat.v = bat.v + (self.best_gen - bat.gen) * bat.freq
                # tmp_gen = bat.gen + bat.v
                
                ## ---> chạy thử cách 1  =================================================
                # tmp_gen = bat.gen.copy()
                
                ## ---> chạy thử cách 2 ============================================
                bat.freq = np.random.randint(self.number_domain-2)
                tmp_gen = PMX_len(self.best.gen, bat.gen, bat.freq)
                print(tmp_gen)

                ### local search
                rnd = np.random.random_sample()
                if rnd > self.r[i]:
                    rand_best = np.random.randint(self.len_top)
                    tmp_gen = deepcopy(self.top_best[rand_best].gen)

                    # swap domain
                    for i in range(int(10*self.A[i])):
                        d1, d2 = np.random.choice(self.order_domain, 2)
                        # d1, d2 = np.random.choice(self.list_domain, 2)
                        tmp_gen[d1], tmp_gen[d2] = tmp_gen[d2], tmp_gen[d1]
                        
                    ## insert 
                    # for i in range(max(1, int(8*self.A[i]))):
                    #     d1, d2 = np.random.choice(self.order_domain, 2)
                    #     # d1 = np.random.randint(self.number_domain)
                    #     # d2 = np.random.randint(self.number_domain)
                    #     # while d1 == d2:
                    #     #     d2 = np.random.randint(self.number_domain)
                    #     node = tmp_gen.pop(d1)
                    #     tmp_gen.insert(d2, node)


                    ## increase and decrease 2 node
                    # for i in range(self.number_domain):
                    #     delta = np.random.uniform(0.3)
                    #     if len(self.domain[i]) > 1:
                    #         node1, node2 = np.random.choice(self.domain[i], 2)
                    #         tmp_gen[node1] += delta
                    #         tmp_gen[node2] -= delta

                ## new fitness
                # path_domain = self.convert_gen_domain(tmp_gen)
                # subgraph = self.gen_subgraph(path_domain)
                # Fnew = idpc_dijkstra(subgraph, self.s, self.t)
                
                subgraph = self.gen_subgraph_new(bat.gen)
                Fnew = idpc_dijkstra(subgraph, self.s, self.t)
                
                ### check to update solution if better and random
                rnd = np.random.random_sample()
                if Fnew <= bat.fitness and rnd < self.A[i]:
                    bat.gen = tmp_gen
                    bat.fitness = Fnew
                    self.pop.append(bat)

                    # increase ri, decrease Ai
                    self.A[i] *= anpha
                    self.r[i] = self.r0 * (1 - math.exp(-gamma*(t/self.N_Gen)))

                ## update best
                if Fnew <= self.best.fitness:
                    self.best = deepcopy(bat)
                     
            self.get_top_best()
            track_best.append(self.best.fitness)
        # plt.plot([i for i in range(len(track_best))], track_best)
        # plt.show()
    
    
    def test_all(self):
        all_case = list(itertools.permutations([i for i in range(10)]))
        file = open('all_case.txt', 'w')
        # hamming
        def hamming_distance(a, b):
            d = 0
            for i in range(len(a)):
                if a[i] != b[i]:
                    d += 1
            return d

        # max_sim
        def max_sim(a, b):
            n = len(a)
            max_d = 0
            for i in range(n):
                index_b = b.index(a[i])
                len_max = min(n-index_b, n-i)
                d = 1
                for c in range(1, len_max):
                    if a[i+c] == b[index_b+c]:
                        d += 1
                    else:
                        if max_d < d:
                            max_d = d
                        break
            return max_d
        
        for case in all_case:
            # if case.index(1) > case.index(0):
            #     continue
            subgraph = self.gen_subgraph(case)
            Fnew = idpc_dijkstra(subgraph, self.s, self.t)
            if Fnew != 999999:
                file.write(str(case) + '-' + str(Fnew) + '\n')
        
        file.close()
    
    def test_case(self):
        case = [1, 2, 0, 4, 6, 5, 8, 9, 3, 7]
        subgraph = self.gen_subgraph(case)
        Fnew = idpc_dijkstra(subgraph, self.s, self.t)
        print('Fnew', Fnew)
            


########################################
########################################
########################################
if __name__ == '__main__':
    directory = 'Project3/result/{}_{}'.format('test', 'insert')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for file_path in glob.glob('Project3\data\Data_IDPCDU_Nodes\*'):
        name = file_path.split('\\')[-1]
        number_node = int(name.split('_')[2])
        if number_node > 500: 
            continue
        # if number_node < 300: 
        #     continue
        # if name != 'idpc_ndu_102_10_834.txt':
        # if name != 'idpc_ndu_202_22_2341.txt':
        # if name != 'idpc_ndu_52_6_204.txt':
            # continue
        print(name)
        
        
        file = open('{}/result_paper_{}.txt'.format(directory, name), 'w')
        node, domains, w, adjacent_list = read_file(file_path)
        nodes = [i for i in range(node)]

        # domains, nodes, w, s, t, NP, N_Gen, A, r, Qmin, Qmax
        lis = []
        start = time.time()
        for i in range(30):
            print('case:', i)
            Algorithm = BatAlgorithm(domains=domains,
                                    nodes=nodes,
                                    w=np.array(w),
                                    adjacent_list=adjacent_list,
                                    s=0,
                                    t=node-1,
                                    NP=200,
                                    N_Gen=250,
                                    A=0.7,
                                    r=0.6,
                                    Qmin=0.0,
                                    Qmax=2.0
                                    )
            
            # Algorithm.move_bat()
            Algorithm.test_all()
            # Algorithm.test_case()
            # dijkstra(Algorithm.w, Algorithm.s, Algorithm.t)
            # print(idpc_dijkstra(Algorithm.w, Algorithm.s, Algorithm.t))
            
            lis.append(Algorithm.best.fitness)
        print('result:', lis)
        print('AVG:', np.mean(lis))
        print('STF:', np.std(lis))
        print('BF:', min(lis))
        
        file.write(name + '\n')
        file.write('AVG:'+str(np.mean(lis)) + '\n')
        file.write('STF:' + str(np.std(lis)) + '\n')
        file.write('BF:' + str(min(lis)) + '\n')
        file.write('\n')
        print(time.time() -start)
        file.close()
