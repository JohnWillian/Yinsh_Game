# IMPORTS AND CONSTANTS#
from collections import deque
import time, random, re

import numpy as np
from Yinsh.yinsh_model import YinshGameRule 
from copy import deepcopy

GAMMA = 0.9
THINKTIME = 0.9
center = (5,5)
#FUNCTION INPLEMENT

class myAgent():
    def __init__(self, id):
        self.id = id
        self.round = 0
        self.gameRule = YinshGameRule(2)

    
    # Generates actions from this state.
    def GetActions(self, state):
        return self.gameRule.getLegalActions(state, self.id)
    
    # Carry out a given action on this state and return True if reward received.
    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.gameRule.generateSuccessor(state, action, self.id)
        result = state.agents[self.id].score - score
        return result
    
    def TransformAsString(self,state):
        str_format="".join(map(str,state.board))
        return re.sub(r'\D',"",str_format)

    def Exploit (self, state, next_actions):
        next_state = deepcopy(state)
        cur_action = random.choice(next_actions)
        reward = self.DoAction(next_state,cur_action)
        next_actions = self.GetActions(next_state)

        return next_actions, cur_action, reward, next_state

    def calculate(self, a , b):
        y1 = a[0]
        x1 = a[1]
        y2 = b[0]
        x2 = b[1]
        deta_x = x1 - x2
        deta_y = y1 - y2
        result = 0
        if deta_x * deta_y <= 0:
            result = max(abs(deta_x),abs(deta_y))
        else:
            result = abs(deta_y)+abs(deta_x)
        return result

    def first_strategy(self, state, actions):
        
        index = self.gameRule.current_agent_index

        board = state.board
        min_toCenter = 51
        min_toOpp = 0
        # max_toCenter = 0
        max_toOpp = 0
        opp_id = 1 - self.id
        opp_pos = state.ring_pos[opp_id]
        result_action = random.choice(actions)
        pos = result_action["place pos"]

        for i in range(11):
            for j in range(11):
                if board[(i,j)] == 0:
                    a = list((i,j))
                    disToC = self.calculate(a,(center))
                    if disToC < min_toCenter:
                        min_toCenter = disToC
                        pos=(i,j)
                    # for ringOfopp in opp_pos:

                    #     disToOpp = self.calculate(a,(ringOfopp))
                    #     if disToOpp > min_toOpp or (disToC < min_toCenter and disToOpp==min_toOpp):
                    #         # or (disToC < min_toCenter and disToOpp==min_toOpp)
                    #         min_toCenter = disToC
                    #         min_toOpp = disToOpp
                    #         pos = (i,j)
                        # else:
                        #     if disToOpp > max_toOpp and disToC < min_toCenter:
                        #         max_toCenter = disToC
                        #         max_toOpp = disToOpp
                        #         pos = (i,j)
        result_action["place pos"] = pos
        
        return result_action

    # Perform MCTS
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        visited_state = dict()
        num_visited = dict()
        str_rootstate = self.TransformAsString(rootstate)
        result_action = random.choice(actions)

        self.round+=1
        if self.round<=5:
            result_action = self.first_strategy(rootstate,actions)
            print("round_5")
            # return result_action
        else:
            while time.time()-start_time < THINKTIME:
                '''Inital all varient.
                    reward can be 1,0,-1;

                '''
                # print(self.round)


                    # return result_action
                state = deepcopy(rootstate)
                next_actions = actions
                str_state = str_rootstate
                queue = deque([str_state])
                
                reward = 0
                length = 0
                selected_action = None

                # Combine select and expand
                while str_state in visited_state and reward == 0 and len(next_actions)>0:
                        if time.time()-start_time >= THINKTIME:
                            # print("MCTS",length)
                            return result_action
                        next_actions, cur_action,reward, next_state = self.Exploit(state, next_actions)

                        if not selected_action:
                            selected_action = cur_action
                        str_state = self.TransformAsString(next_state)
                        queue.append(str_state)
                        state = next_state

                        
                        

                # Simulate
                if reward == 0 and len(next_actions)>0:
                    
                    while reward == 0 and length < 13 and len(next_actions)>0:
                        # time out
                        
                        if time.time()-start_time >= THINKTIME:
                            print("MCTS",length)
                            return result_action
                        length += 1
                        next_actions, cur_action,reward, next_state = self.Exploit(state, next_actions)
                        state = next_state

                        # print(length)
                        
                else:
                    pass

                # Backpropagation
                simulate_value = reward * (GAMMA ** length)    
                while len(queue)>0 and time.time()-start_time < THINKTIME:
                    pre_state = queue.pop()
                    if pre_state in visited_state:
                        # If the pre_state is the root, 
                        # then we set the action of root as selected action 
                        if pre_state == str_rootstate and simulate_value > visited_state[pre_state]:
                            result_action = selected_action

                        # Update the visited_state of each state
                        visited_state[pre_state] = max(visited_state[pre_state],simulate_value)
                        num_visited[pre_state] += 1
                    
                    else:
                        visited_state[pre_state] = simulate_value
                        num_visited[pre_state] = 1
                    simulate_value *= GAMMA

        return result_action
