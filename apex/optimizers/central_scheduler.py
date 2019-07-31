#N = 8
def generate_schedule_list(N):

    temp_list = [[(j,i) for i in range(N) if i!=j] for j in range(N)]
    key_list = [[0 for i in range(N-1)] for j in range(N)]
    schedule_list = [[] for i in range(2*(N-1))]
    worker_schedule = [[] for i in range(N)]
    for t in range(2*(N-1)):
        temp_key = [0 for i in range(N)]
        for pos in range(N):
            if temp_key[pos] == 1:
                continue
            else:
                for j in range(N-1):
                    rec = temp_list[pos][j][1]
                    if key_list[pos][j] == 0 and temp_key[rec]==0:
                        
                        key_list[pos][j] = 1
                        temp_key[rec] = 1
                        temp_key[pos] = 1
                        schedule_list[t].append([temp_list[pos][j][0],temp_list[pos][j][1]])
                        #print(temp_list[pos][j][0],' ',temp_list[pos][j][1],'|',end='')
                        break
        #print('\n')
    for worker in range(N):
        for t in range(2*(N-1)):
            for pair in schedule_list[t]:
                if pair[0] == worker:
                    worker_schedule[worker].append([pair[1],0])
                if pair[1] == worker:
                    worker_schedule[worker].append([pair[0],1])
    return worker_schedule
#for i in range(len(schedule_list)):
 #   print(schedule_list[i])
#for i in range(len(worker_schedule)):
 #   print(worker_schedule[i])
#print(schedule_list)
#print(worker_schedule)
    #return schedule_list
    
#print(schedule_list)
