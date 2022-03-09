import numpy as np
from copy import deepcopy


class Individual:
    def __init__(self, a):
        self.path = a

def get_min_v(queue, best_dist):
    best = 999999
    min_v = -1
    for v in queue:
        if best_dist[v] < best:
            best = best_dist[v]
            min_v = v

    return min_v


def dijkstra(weight_matrix, start, end):
    n = len(weight_matrix)
    path = []
    pre = [0  for i in range(n)]
    best_dist = [999999 for i in range(n)]
    best_dist[0] = 0

    visited = []
    queue = [start]
    while queue:
        current = get_min_v(queue, best_dist)
        # print(queue)
        if current == end:
            break
        for i in range(n):
            if weight_matrix[current][i] == 0:
                continue
            if i in visited:
                continue
            new_cost = best_dist[current] + weight_matrix[current][i]
            if new_cost < best_dist[i]:
                best_dist[i] = new_cost
                if i not in queue:
                    queue.append(i)
        queue.remove(current)
        visited.append(current)

    return best_dist[end]


def idpc_dijkstra(weight_matrix, start, end):
    n = len(weight_matrix) # number node
    
    path = []
    pre = [0  for i in range(n)]
    best_dist = [999999 for i in range(n)]
    best_dist[start] = 0

    code_queue = [0 for i in range(n)] # to faster than use check in queue
    not_visited = [i for i in range(n)] # to faster than use check in visited
    queue = [start]
    code_queue[start] = 1
    while queue:
        current = get_min_v(queue, best_dist)
        if current == end:
            break
        for i in not_visited:
            if weight_matrix[current][i] == 0:
                continue
            new_cost = best_dist[current] + weight_matrix[current][i]
            if new_cost < best_dist[i]:
                best_dist[i] = new_cost
                # if i not in queue:
                #     queue.append(i)
                if code_queue[i] == 0:
                    queue.append(i)
                    code_queue[i] = 1
        queue.remove(current)
        # code_queue[current] = 2
        not_visited.remove(current)

    return best_dist[end]


def PMX_len(arr1, arr2, length):
    n = len(arr1) # number of gen
    p1 = np.random.randint(n-length)
    p2 = p1 + length-1
    p2 += 1
    
    offspring2 = deepcopy(arr1)
    offspring1 = deepcopy(arr2)
    mid1 = arr1[p1: p2]
    mid2 = arr2[p1: p2]

    ## giư lại đoạn giữa
    offspring1[p1:p2], offspring2[p1:p2] = arr1[p1:p2], arr2[p1:p2] 
    
    for i in range(p1, p2):
        save_point = arr2[i]
        point2 = arr2[i]
        index = i
        
        if point2 in mid1:
            continue
        
        while p1 <= index <= p2 -1:
            point1 = arr1[index] # điểnm bên trên
            index = arr2.index(point1)
            point2 = arr2[index]
            
        offspring1[index] = save_point
        
    return offspring1


def SPX_len(arr1, arr2, length):
    n = len(arr1)
    p = np.random.randint(n-length)
    offspring1 = deepcopy(arr1)
    
    tmp_p = 0
    for i in range(p, n):
        for j in range(tmp_p, n):
            if arr2[j] in offspring1[:i]:
                continue
            offspring1[i] = arr2[j]
            tmp_p = j
            break
    
    return offspring1

