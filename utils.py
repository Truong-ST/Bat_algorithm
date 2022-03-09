import numpy as np


def read_file(file_path):
    file = open(file_path, 'r')
    node, number_domain = [int(x) for x in file.readline().split(' ')]
    domains = []
    for i in range(number_domain):
        a = [int(x)-1 for x in file.readline().split(' ')]
        domains.append(a)
    number_edge = int(file.readline())
    weight = [[0 for i in range(node)] for j in range(node)]
    adjacent_list = [[] for i in range(node)]
    for i in range(number_edge):
        s, e, w = [int(x)-1 for x in file.readline().split(' ')]
        weight[s][e] = w + 1
        adjacent_list[s].append(e)
    file.close()
    
    return node, domains, weight, adjacent_list


def write_dot(file_path):
    file = open(file_path, 'r')
    file_dot = open(file_path.replace('.txt', '.dot'), 'w')
    file_dot.write('digraph map {\n')
    node, number_domain = [int(x) for x in file.readline().split(' ')]
    domains = []
    color=['red', 'blue', 'green', 'yellow', 'brown', 'purple']
    for i in range(number_domain):
        a = [int(x)-1 for x in file.readline().split(' ')]
        for x in a:
            file_dot.write('\t{x}[color={c},style=filled]\n'.format(x=x, c=color[i]))
        domains.append(a)
    number_edge = int(file.readline())
    for i in range(number_edge):
        s, e, w = [int(x)-1 for x in file.readline().split(' ')]
        file_dot.write('\t{s} -> {e}[label={w}]\n'.format(s=s,e=e,w=w+1))
    
    file_dot.write('}')
    file.close()
    file_dot.close()


def hamming_distance(x1, x2):
    d = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            d += 1
    return d


def distance(x1, x2):
    return np.linalg.norm(x1-x2)


def insert_function(x):
    p1 = np.random.randint(len(x))
    p2 = np.random.randint(len(x))
    value = x.pop(p1)
    x.insert(p2, value)
    
    return x
    
    
def exchange_funtion(x):
    p1 = np.random.randint(len(x))
    p2 = np.random.randint(len(x))
    x[p1], x[p2] = x[p2], x[p1]
    
    return x


    