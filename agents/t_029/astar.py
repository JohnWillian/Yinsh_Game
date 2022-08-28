# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Xinzhou Jiang
# Date:    24/05/2022
# Purpose: Implements an A_star search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time,random,re,heapq
from Yinsh.yinsh_model import YinshGameRule 
from copy import deepcopy
import sys 
sys.path.append('agents/t_029/')

THINKTIME = 0.65
diag_bound = [(2,6),(1,7),(0,7),(0,7),(0,7),(0,6),(1,5)]


# PriorityQueue Class
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

# myAgent Class
class myAgent():
    def __init__(self, _id):
        # read combinations
        comb =open("agents/t_029/combination.csv").readlines()
        comblines = []
        for line in comb:
            comblines.append(line.strip().split(","))
        self.hValue = dict()
        for item in comblines:
            self.hValue[item[0]+item[1]] = int(item[2])
        self.id = _id
        self.round=0
        self.game_rule = YinshGameRule(2) #2 players
    
    def get_candidate_actions(self,state):
        legal_actions=self.game_rule.getLegalActions(state, self.id)
        if self.round>5:
            #we should start from the center to increase the winning rate
            num=int(len(legal_actions)) #index:0-num-1
            candidates=[]
            '''
            if num%2==0:
                idx=int(num/2)
            else:
                idx=int((num-1)/2) 
            candidates.append(legal_actions[idx])
            sign=1
            for i in range(1,num): #2-1-3-0; 1-0-2
                sign=-sign
                idx=idx+sign*num
                candidates.append(legal_actions[idx])
            '''
            idx1=int(num/4)
            idx2=int(3*num/4)
            if num>3:
                candidates=legal_actions[idx1:idx2]+legal_actions[0:idx1]+legal_actions[idx2:num]
            else:
                return legal_actions
        else:  
            #we should start from the center to increase the winning rate
            num=len(legal_actions) #index:0-num-1
            candidates=[]
            if num%2==0:
                idx=int(num/2)
            else:
                idx=int((num-1)/2) 
            candidates.append(legal_actions[idx])
            candidates.append(legal_actions[idx-1])
            candidates.append(legal_actions[idx+1])

        return candidates

    # calculate whether it can get reward sfter this action
    def ifEarn(self, state, action):
        score = state.agents[self.id].score
        nextState = self.game_rule.generateSuccessor(state, action, self.id)
        return nextState.agents[self.id].score > score

    # Serach actions that lead to a reward. If there is, return the first one; if not, return the action that leads to min h.
    def SelectAction(self, actions, rootstate):
        self.round+=1
        start_time = time.time()
        #count = 0 
        queue = PriorityQueue()
        queue.push((deepcopy(rootstate),0,100,[]) ,0) # initialize the priority queue: state+path
        best_key = ''
        best_gn=100 #g+h
        best_path={}
        visited_key=[]
        #A* algorithm
        while (not queue.isEmpty()) and ((time.time()-start_time) < THINKTIME):
            #count += 1
            state, cost, gn, path = queue.pop() #This is a priority queue, each time it returns the smallest gn
            key = re.sub(r'\D',"","".join(map(str,state.board)))
            if ((best_key=='') or (key not in visited_key)): #if it is the beginning or gn<best_g
                if (gn < best_gn):
                    best_key=key
                    best_gn = gn
                    #print("update_best")
                    if len(path)>0:
                        best_path=path[0]
                visited_key.append(key)
                new_actions = self.get_candidate_actions(state)
                for a in new_actions: 
                    next_state = deepcopy(state) 
                    next_path  = path + [a] 
                    earn = self.ifEarn(next_state, a) # whether it can form a string of five pieces and earn a reward
                    if earn:
                        print('path found:', next_path)
                        #print("astar",count)
                        return next_path[0] # If this action was rewarded, return the initial action that led there.
                    else:
                        gn= cost + 1 + self.CalHeuristic(next_state.board)
                        queue.push((next_state, cost + 1,gn, next_path),gn)
        #print("astar",count)
        if ((time.time()-start_time) >= THINKTIME):
            #print("exceed")
            return best_path
            #return random.choice(actions) # if it exceeds the time limit, return a random action
        else:
            #print("best")
            return best_path
            
    def CalHeuristic(self,graph):
        min_value = 51
        hori_value=51
        verti_value=51
        diag_value=51
        for i in range(1,10): # horizontal and vertical, 1-9
            start = max(0,5-i)
            end = min(7,12-i)
            for j in range(start,end):
                horizontal = str(self.id)+str(graph[(i,j)])+str(graph[(i,j+1)])+str(graph[(i,j+2)])+str(graph[(i,j+3)])+str(graph[(i,j+4)])
                vertical = str(self.id)+str(graph[(j,i)])+str(graph[(j+1,i)])+str(graph[(j+2,i)])+str(graph[(j+3,i)])+str(graph[(j+4,i)])
                if horizontal in self.hValue:
                    hori_value = self.hValue[horizontal]
                if vertical in self.hValue:
                    verti_value = self.hValue[vertical]
                min_value = min(min_value, hori_value,verti_value) #update min_value
        for i in range(0,7):
            start, end = diag_bound[i]
            for j in range (start,end):
                diagonal = str(self.id)+str(graph[(i,j)])+str(graph[(i-1,j+1)])+str(graph[(i-2,j+2)])+str(graph[(i-3,j+3)])+str(graph[(i-4,j+4)])
                if diagonal in self.hValue:
                    diag_value = self.hValue[diagonal]
                min_value = min(min_value,diag_value) #update min_value
        return min_value

    
# END FILE -----------------------------------------------------------------------------------------------------------#
