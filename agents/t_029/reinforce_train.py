# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Yu Zhang
# Date:    24/05/2022
# Purpose: Implements an Reinforcement Learning trainer agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import csv
import time
import random

from regex import R
from Yinsh.yinsh_model import YinshGameRule
from copy import deepcopy
import numpy as np
import sys
sys.path.append('agents/t_029/')

THINKTIME = 0.95

DECAY = 0.99
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor

# python yinsh_runner.py --teal agents.t_029.reinforce_train --magenta agents.example_bfs -q -m 3
# python yinsh_runner.py --teal agents.example_bfs --magenta agents.t_029.reinforce_player -p
# python yinsh_runner.py --teal agents.t_029.reinforce_train --magenta agents.t_029.reinforce_train -q -m 3
# python yinsh_runner.py --teal agents.t_029.reinforce_player --magenta agents.t_029.reinforce_competitor -p
# python yinsh_runner.py --teal agents.t_029.reinforce_competitor --magenta agents.t_029.reinforce_player -p

class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.game_rule = YinshGameRule(2)
        self.weight = [0, 0, 0, 0, 0, 0]
        self.init_weight = [0, 0, 0]
        # self.hValue = 51
        self.round = 0
        self.EPSIL = 0.5
        self.EPSIL1 = 0.2

       # read combinations
        comb = open("agents/t_029/combination.csv").readlines()
        comblines = []
        for line in comb:
            comblines.append(line.strip().split(","))
        self.h_value = dict()
        for item in comblines:
            self.h_value[item[0]+item[1]] = int(item[2])

        # read weights
        weights = open("agents/t_029/weights.csv").readlines()
        weightslines = weights[0].strip().split(",")
        weightslines = [float(i) for i in weightslines]
        self.weight = weightslines

        # read initial weights
        init_weights = open("agents/t_029/init_weights.csv").readlines()
        init_weightslines = init_weights[0].strip().split(",")
        init_weightslines = [float(i) for i in init_weightslines]
        self.init_weight = init_weightslines
        
        # print(self.weight)

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action):
        score = state.agents[self.id].score
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - score

    def getHeuristicValue(self, board):
        max_h_value = 0
        if self.id == 0:
            ring = "1"
        else:
            ring = "3"
        for i in range(11):
            for j in range(11):
                if j + 5 < 11:
                    horizontal = str(
                        self.id) + "".join([str(board[(i, j+k)]) for k in range(5)])
                    vertical = str(
                        self.id) + "".join([str(board[(j+k, i)]) for k in range(5)])
                    v1, v2 = 0, 0
                    if horizontal in self.h_value:
                        if horizontal.count(ring) > 0:
                            v1 = 5
                        v1 += 5 - self.h_value[horizontal]
                    if vertical in self.h_value:
                        if vertical.count(ring) > 0:
                            v2 = 5
                        v2 += 5 - self.h_value[vertical]
                    max_h_value = max(max_h_value, v1, v2)

            for l in range(11):
                if l + 5 < 11 and i - 5 > 0:
                    diagonal = str(
                        self.id) + "".join([str(board[(i-k, l+k)]) for k in range(5)])
                    v3 = 0
                    if diagonal in self.h_value:
                        if diagonal.count(ring) > 0:
                            v3 = 5
                        v3 += 5 - self.h_value[diagonal]
                    max_h_value = max(max_h_value, v3)
        return max_h_value

    def getContinuous(self, board, player):
        v1, v2, v3 = 0, 0, 0
        res = 0
        count = 0
        piece = str(player)
        ring = str(player - 1)
        max_val = 0
        for i in range(11):
            for j in range(11):
                if j + 5 < 11:
                    horizontal = str(
                        self.id) + "".join([str(board[(i, j+k)]) for k in range(5)])
                    vertical = str(
                        self.id) + "".join([str(board[(j+k, i)]) for k in range(5)])

                    if horizontal.count(piece) > 3:
                        if horizontal.count(ring) > 1:
                            v1 += 5
                        v1 += 1
                    if vertical.count(piece) > 3:
                        if vertical.count(ring) > 1:
                            v2 += 5
                        v2 += 1
                    max_val = max(max_val, v1, v2)
                    count += 12
                else:
                    break

            for l in range(11):
                if l + 5 < 11 and i - 5 > 0:
                    diagonal = str(
                        self.id) + "".join([str(board[(i-k, l+k)]) for k in range(5)])
                    if diagonal.count(piece) > 3:
                        if diagonal.count(ring) > 1:
                            v3 += 5
                        v3 += 1
                    max_val = max(max_val, v3)
                    count += 6
                else:
                    break
        if player == 2:
            res = max_val / count
        else:
            res = (count - max_val) / count
        return res
    
    def initGetContinuous(self, board, player):
        res = 0
        count = 1
        player = str(player)
        for i in range(11):
            for j in range(11):
                if j + 5 < 11:
                    horizontal = str(
                        self.id) + "".join([str(board[(i, j+k)]) for k in range(5)])
                    vertical = str(
                        self.id) + "".join([str(board[(j+k, i)]) for k in range(5)])

                    if horizontal.count(player) >= 3:
                        res += 1
                    if vertical.count(player) >= 3:
                        res += 1
                count += 1

            for l in range(11):
                if l + 5 < 11 and i - 5 > 0:
                    diagonal = str(
                        self.id) + "".join([str(board[(i-k, l+k)]) for k in range(5)])
                    if diagonal.count(player) >= 3:
                        res += 1
                count += 1
        res = res / count
        return res

    def getFeatures(self, state, action):
        next_state = deepcopy(state)
        total_steps = 51
        # Identify players
        if self.id == 0:
            player1 = 2
            player2 = 4
        else:
            player1 = 4
            player2 = 2

        # Feature 1: Get h value for current state
        f1 = self.getHeuristicValue(next_state.board) / 10
        # Feature 2: The case of 4 consecutive pieces of the same color of a sequence
        f2 = self.getContinuous(next_state.board, player1)
        # Feature 3: Block the opponent's path when they are about to win
        f3 = self.getContinuous(next_state.board, player2)
        # Feature 4: Get next action score
        f4 = self.DoAction(next_state, action) / 3
        # Feature 5: Get the number of pieces on the board
        temp_board = np.array(next_state.board).flatten()
        checkerboard = "".join([str(i) for i in temp_board])
        f5 = checkerboard.count(str(player1)) / total_steps
        # Feature 6: Get the number of pieces on the board (opponet)
        f6 = checkerboard.count(str(player2)) / total_steps
        
        return [f1, f2, f3, f4, f5, f6]

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        Q_value = 0
        n = len(features)
        for i in range(n):
            Q_value += features[i] * self.weight[i]
        # print("features:", features)
        # print("Q_value:", Q_value)
        return Q_value

    def getCenterDist(self, state, action):
        board = state.board
        min_dist = 10
        pos = action["place pos"]
        center = (5, 5)
        dist = abs(pos[0] - center[0]) + abs(pos[1] - center[1])

        return (11 - dist) / 11

    def getOppoDist(self, state, action):
        if self.id == 0:
            o_id = 1
        else:
            o_id = 0
        pos = action["place pos"]
        min_dist = 20
        o_rings = state.ring_pos[o_id]
        for r in o_rings:
            dist = abs(pos[0] - r[0]) + abs(pos[1] - r[1])
            min_dist = min(dist, min_dist)

        return (21 - min_dist) / 21

    def getInitFeatures(self, state, action):
        if self.id == 0:
            player = 1
        else:
            player = 3
        f1 = self.getCenterDist(state, action)
        f2 = self.getOppoDist(state, action)
        f3 = self.initGetContinuous(state.board, player)

        return [f1, f2, f3]

    def initialAction(self, state, action):
        features = self.getInitFeatures(state, action)
        Q_value = 0
        n = len(features)
        for i in range(n):
            Q_value += features[i] * self.init_weight[i]
        return Q_value

    def SelectAction(self, actions, game_state):
        self.round += 1
        max_Q_value = -float('inf')
        max_Q_value1 = -float('inf')
        init_time = time.time()
        best_action = random.choice(actions)

        if self.round <= 5:
            n = len(actions)
            i = 0
            # Epsilon-decreasing strategy
            if random.uniform(0, 1) < 1 - self.EPSIL1:
                while time.time() - init_time < THINKTIME and i < n:
                    Q_value = self.initialAction(
                        deepcopy(game_state), actions[i])
                    if Q_value > max_Q_value:
                        max_Q_value = Q_value
                        best_action = actions[i]
                    i += 1
                self.EPSIL1 = DECAY * self.EPSIL1
            else:
                max_Q_value = self.initialAction(
                    deepcopy(game_state), best_action)

            next_state = (deepcopy(game_state))
            reward = self.DoAction(next_state, best_action)
            next_actions = self.GetActions(next_state)
            max_next_Q_value = -float('inf')

            m = len(next_actions)
            j = 0
            while time.time() - init_time < THINKTIME and j < m:
                Q_value = self.initialAction(
                    deepcopy(next_state), next_actions[j])
                max_next_Q_value = max(max_next_Q_value, Q_value)
                j += 1

            features = self.getInitFeatures(deepcopy(game_state), best_action)
            delta = reward + GAMMA * max_next_Q_value - max_Q_value
            k = len(features)
            j = 0
            while time.time() - init_time < THINKTIME and j < k:
                self.init_weight[j] += ALPHA * delta * features[j]
                j += 1

            csv_file = open("agents/t_029/init_weights.csv",
                            'w', encoding='utf-8', newline='')
            write = csv.writer(csv_file)
            write.writerow(self.init_weight)
            csv_file.close()
            return best_action

        # Epsilon-decreasing strategy
        if random.uniform(0, 1) < 1 - self.EPSIL:
            for action in actions:
                if time.time() - init_time > THINKTIME:
                    print("Time Out")
                    break
                Q_value = self.getQValue(deepcopy(game_state), action)
                if Q_value > max_Q_value1:
                    max_Q_value1 = Q_value
                    best_action = action
            self.EPSIL = DECAY * self.EPSIL
        else:
            max_Q_value1 = self.getQValue(deepcopy(game_state), best_action)

        next_state = (deepcopy(game_state))
        reward = self.DoAction(next_state, best_action)

        next_actions = self.GetActions(next_state)
        max_next_Q_value = -float('inf')

        m = len(next_actions)
        i = 0
        while time.time() - init_time < THINKTIME and i < m:
            Q_value = self.getQValue(deepcopy(next_state), next_actions[i])
            max_next_Q_value = max(max_next_Q_value, Q_value)
            i += 1

        features = self.getFeatures(deepcopy(game_state), best_action)
        delta = reward + GAMMA * max_next_Q_value - max_Q_value1

        k = len(features)
        j = 0
        while time.time() - init_time < THINKTIME and j < k:
            self.weight[j] += ALPHA * delta * features[j]
            j += 1

        csv_file = open("agents/t_029/weights.csv", 'w',
                        encoding='utf-8', newline='')
        write = csv.writer(csv_file)
        write.writerow(self.weight)
        csv_file.close()

        print(time.time() - init_time)
        return best_action
