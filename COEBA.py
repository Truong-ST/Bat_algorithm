import random
import math
import time
import numpy as np
# import matplotlib.pyplot as plt
from utils import *
from algorithm import *
import glob


class Bat:
    def __init__(self, gen, v, freq):
        self.gen = gen  # [priority_i for i in V]
        self.path = []
        self.v = v
        self.freq = freq
        self.fitness = 0

    def update_fitness(self, w, s):
        self.fitness = self.func(self.path, w, s)


class COEBA():
    def __init__(self, domains, nodes, w, adjacent_list, s, t, NP, N_Gen, A, r, Qmin, Qmax):
        ## process graph
        self.w = w  # weight
        self.nodes = nodes  # node
        self.s = s  # start
        self.t = t  # end
        self.domain = domains  # domain correspond node
        self.number_node = len(nodes)
        self.number_domain = len(self.domain)
        self.adjacent_list = adjacent_list
        self.k = 5

        ## hyperparameter
        self.NP = NP  # population size
        self.N_Gen = N_Gen  # generations
        self.A = A  # loudness
        self.r0 = r  # pulse rate
        self.r = 0
        self.Qmin = Qmin  # frequency min
        self.Qmax = Qmax  # frequency max

        self.total_priority_domain = [1 for i in range(self.number_domain)]
        self.pop = [Bat([0], [0], 0) for i in range(self.NP)]

        ## best
        self.best = self.pop[0].path  # x
        self.best_gen = self.pop[0].gen
        self.f_min = self.pop[0].fitness

    def best_bat(self):
        j = 0
        for i in range(self.NP):
            if self.pop[i].fitness < self.pop[j].fitness:
                j = i
        self.best = self.pop[j].path
        self.f_min = self.pop[j].fitness

    def random_devide_const(self, const, len_list):
        arr = np.full(len_list, const/len_list)
        for i in range(len_list):
            arr[i] += np.random.uniform(-const/3/len_list, const/3/len_list)
        return arr

    def devide_domain(self):
        """
        devide priority in domain by const
        """
        priority = np.zeros(self.number_node)
        for i in range(self.number_domain):
            number_node_domain = len(self.domain[i])
            priority_domain = self.random_devide_const(
                number_node_domain, number_node_domain)
            for j in range(number_node_domain):
                priority[self.domain[i][j]] = priority_domain[j]
        priority[self.t] += 1
        return priority

    def convert_gen_x(self, gen):
        """
        decode gen to path
        """
        current = self.s
        x = []
        while current != self.t and len(x) != self.number_node - 1:
            next_node = -1
            better_priority = -1
            for node in self.adjacent_list[current]:
                if node in x:
                    continue
                if gen[node] > better_priority:
                    better_priority = gen[node]
                    next_node = node
            current = next_node
            x.append(next_node)
            if next_node == -1:
                break

        return x

    # Fitness function
    def fitness_path(self, path):
        # distance
        if path[-1] == -1:
            return 9999999
        sum = self.w[self.s][path[0]]
        for i in range(len(path)-1):
            sum += self.w[path[i]][path[i+1]]

        # domain
        # remain_domain = [i for i in range(self.number_domain)]
        # for node in path:
        #     for i in range(self.number_domain):
        #         if node in self.domain[i]:
        #             if i in remain_domain:
        #                 remain_domain.remove(i)
        #             break
        # sum += len(remain_domain)*100

        return sum


    def init_bat(self):
        # !! devide total priority by domain
        for bat in self.pop:
            bat.freq = 0
            bat.v = np.zeros(self.number_node)
            bat.gen = self.devide_domain()  # sum of priority node in domain is const
            # bat.gen = [i for i in range(self.number_node)]
            random.shuffle(bat.gen)
            bat.path = self.convert_gen_x(bat.gen)
            bat.fitness = self.fitness_path(bat.path)

        self.best_bat()

    def move_bat(self):
        self.init_bat()
        K = [[] for i in range(self.k)]
        for i, bat in enumerate(self.pop):
            a = i % self.k
            K[a].append(bat)

        # New
        track_best = []
        for t in range(self.N_Gen):
            # print('generation: ' + str(t))
            # for i, bat in enumerate(self.pop):
            for i in range(self.k):
                for bat in K[i]:
                    ## update to near best
                    # bat.freq = self.Qmin + (self.Qmax - self.Qmin) * np.random.uniform(0, 1)
                    # bat.v = bat.v + (self.best_gen - bat.gen) * bat.freq
                    # tmp_gen = bat.gen + bat.v
                    tmp_gen = bat.gen.copy()

                    ## local search (change 1 random node)
                    rnd = np.random.random_sample()
                    if rnd > self.r:
                        ## swap
                        for j in range(self.number_domain):
                            if len(self.domain[j]) > 1:
                                node1, node2 = np.random.choice(self.nodes, 2)
                                tmp_gen[node1], tmp_gen[node2] = tmp_gen[node2], tmp_gen[node1]

                        ## increase and decrease 2 node
                        # for i in range(self.number_domain):
                        #     delta = np.random.uniform(0.3)
                        #     if len(self.domain[i]) > 1:
                        #         node1, node2 = np.random.choice(self.domain[i], 2)
                        #         tmp_gen[node1] += delta
                        #         tmp_gen[node2] -= delta

                    ## new fitness
                    path = self.convert_gen_x(tmp_gen)
                    Fnew = self.fitness_path(path)

                    ## check to update solution if better and random
                    rnd = np.random.random_sample()
                    # if Fnew <= self.f_min and rnd < self.A:
                    if Fnew <= bat.fitness and rnd < self.A:
                        bat.gen = tmp_gen
                        bat.path = path
                        bat.fitness = Fnew

                        # increase ri, decrease Ai
                        anpha = 0.98
                        gamma = 0.98
                        self.A *= anpha
                        self.r = self.r0 * (1-math.exp(-gamma*(t/self.N_Gen)))

                    # update best
                    if Fnew <= self.f_min:
                        self.best = path
                        self.f_min = Fnew
                ## migr
                number_migr = int(0.1 * len(K[i]))
                for i in range(number_migr):
                    rand_group = np.random.randint(self.k)
                    pos_k, pos_rand = np.random.choice([j for j in range(len(K[i]))], 2)
                    K[i][pos_k], K[rand_group][pos_rand] = K[rand_group][pos_rand], K[i][pos_k]
                # Two random individuals are migrated from k to another randomly selected subpopulation (exchange n individuals of k to other)
                # 1 individual is in 10 best
            track_best.append(self.f_min)
        # plt.plot([i for i in range(len(track_best))], track_best)
        # plt.show()


########################################
########################################
########################################
if __name__ == '__main__':
    # file_path = 'E:\HUST\Project3\data\Data_IDPCDU_Nodes\idpc_ndu_52_6_204.txt'
    # file_path = 'E:\HUST\Project3\data\Data_IDPCDU_Nodes\idpc_ndu_102_10_834.txt'
    # file_path = 'E:\HUST\Project3\data\Data_IDPCDU_Nodes\idpc_ndu_152_14_1869.txt'
    
    for file_path in glob.glob('Project3\data\Data_IDPCDU_Nodes\*'):
        name = file_path.split('\\')[-1]
        numer_node = int(name.split('_')[2])
        file = open('Project3\data\\{}.txt'.format(numer_node), 'w')
        if numer_node > 500: 
            continue
        # if name != 'idpc_ndu_102_10_834.txt':
        # if name != 'idpc_ndu_202_22_2341.txt':
        # if name != 'idpc_ndu_52_6_204.txt':
            # continue
        
        node, domains, w, adjacent_list = read_file(file_path)
        nodes = [i for i in range(node)]
        lis = []

        # domains, nodes, w, s, t, NP, N_Gen, A, r, Qmin, Qmax, function
        for i in range(30):
            print('case:', i)
            Algorithm = COEBA(domains=domains,
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
            start = time.time()
            Algorithm.move_bat()
            # print(time.time() - start)

            # for bat in Algorithm.pop:
            #     print(bat.gen)
            # print('best path:', [x+1 for x in Algorithm.best])
            # print('best fitness:', Algorithm.f_min)
            lis.append(Algorithm.f_min)
        print(lis)
        print(name)
        print('AVG:'+str(sum(lis)/ len(lis)))
        print('STF:' + str(np.std(lis)))
        print('BF:' + str(min(lis)))
        file.write('AVG:'+str(sum(lis)/ len(lis)) + '\n')
        file.write('STF:' + str(np.std(lis)) + '\n')
        file.write('BF:' + str(min(lis)) + '\n')
        file.write('\n')
        file.close()

