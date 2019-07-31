w_size = 6
w_list = [ [] for j in range(w_size)]
aux_list = [[0 for i in range(w_size)] for j in range(w_size-1)]
for t in range(w_size - 1):
    for pos in range(w_size):
        rec = (pos + 1 + t)%w_size
        print(pos,rec)
        if aux_list[t][pos] == 0 and aux_list[t][rec] == 0:
            aux_list[t][pos] = 1
            aux_list[t][rec] = 1
            w_list[t].append([pos,rec])

print(w_list)
print(aux_list)          