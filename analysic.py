import numpy 
import itertools
import matplotlib.pyplot as plt


a = list(itertools.permutations([i for i in range(10)]))
# a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# b = [0, 1, 2, 4, 6, 5, 8, 9, 3, 7]
# c = [7, 9, 1, 2, 3, 4, 5, 6, 0, 8]
print(len(a))

#hamming
def hamming_distance(a, b):
    d = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            d += 1
    return d

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

# file = open('all_case.txt', 'r')
# lines = file.readlines()
# file.close()

# data = {}
# for line in lines:
#     a = line[:-1].split('-')
#     data[a[0]] = int(a[1])
    
# new_data = sorted(data.items(), key=lambda x: x[1])

# def new(a):
#     return [int(x) for x in a[1: -1].split(', ')]

# # with open('all_case_sort.txt', 'w') as file:
# #     for k in new_data:
# #         file.write(str(k[0]) + '-' + str(k[1]) + '\n')

# new_data = list(map(lambda x: [new(x[0]), x[1]], new_data))
# # print(new_data)

# hamming_change = [0 for i in range(11)]
# # for i in range(len(new_data)):
# #     for j in range(len(new_data)):
# #         if i == j:
# #             continue
# #         if new_data[i][1] != new_data[j][1]:
# #             hamming_change[max_sim(new_data[i][0], new_data[j][0])] += 1
# # hamming_change = [0, 252043618, 607228822, 238905806, 33219042, 3702154, 361664, 24844, 0, 0, 0]

# # with open('new.txt', 'w') as file:
# #     file.write(str(hamming_change))

# plt.bar([i for i in range(11)], hamming_change)
# plt.show()


            
