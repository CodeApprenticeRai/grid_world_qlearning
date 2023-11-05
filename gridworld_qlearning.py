import numpy as np
import copy

class GridWorldQLearning:
    def __init__(self):
        self.row_MAX = 3
        self.column_MAX = 4
        self.actions_N = 4
        self.state_space = self.init_state_space()
        self.action_space = self.init_action_space()
        self.q = self.init_q_table()

        self.learning_rate = 0.5
        self.discount_factor = 0.25

        self.standard_reward = - 0.08
        self.terminal_reward = {
            (0, 3): 1,
            (1, 3): -1
        }

    def init_state_space(self):
        self.state_space = set()
        for i in range(self.row_MAX):
            for j in range(self.column_MAX):
                self.state_space.add((i, j))
        return self.state_space

    def init_action_space(self):
        self.action_space = set([
            i for i in range(self.actions_N)
        ])
        return self.action_space

    def init_q_table(self):
        self.q = {}
        for state in self.state_space:
            if state not in self.q:
                self.q[state] = {}
            for action in self.action_space:
                # if ((state in self.terminal_reward) or (self.terminal_reward[state] < 0)):
                self.q[state][action] = 0
                # else:
                #     self.q[state] = 1 #this is a prior
        return self.q

    def get_reward(self, state):
        if state not in self.terminal_reward:
            return self.standard_reward
        else:
            return self.terminal_reward[state]

    def get_transition_state(self, state, action):
        # 0-up, 1-down, 2-right, 3-left
        match action:
            case 0:
                if (state[0] == 0):
                    return state
                else:
                    return (state[0]-1, state[1])
            case 1:
                if (state[0]+1 == self.row_MAX):
                    return state
                else:
                    return (state[0]+1, state[1])
            case 2:
                if (state[1] == 0):
                    return state
                else:
                    return (state[0], state[1]-1)
            case 3:
                if (state[1]+1 == self.column_MAX):
                    return state
                else:
                    return (state[0], state[1]+1)


    '''
    Each step update  the q of every possible action in every possible state
    '''

    def update_q_table(self):
        for state in self.state_space:
            for action in self.action_space:
                self.update_q(state, action)
        return self.q

    def update_q(self, state, action):
        q_intmdry1 = 1 - self.learning_rate * self.q[state][action]
        q_intmdry2 = self.discount_factor * self.get_reward(state)
        # max next state q based on action
        max_next_state_q = self.q[state][action]
        for _action in self.action_space:
            state2 = self.get_transition_state(state, _action)
            max_next_state_q = max(max_next_state_q, self.q[state2][_action])
        q_intmdry3 = self.learning_rate * self.discount_factor * max_next_state_q
        self.q[state][action] = sum([q_intmdry1, q_intmdry2, q_intmdry3])
        return self.q[state][action]

    def reset(self):
        self.init_state_space()
        self.init_action_space()
        self.init_q_table()
        return None

    def compute_q_delta(self, new_q_table, old_q_table):
        self.q_delta = {}
        for state in self.q:
            if state not in self.q_delta:
                self.q_delta[state] = {}
            for action in self.q[state]:
                self.q_delta[state][action] = abs(new_q_table[state][action] - old_q_table[state][action])
        return self.q_delta

    def compute_max_q_delta(self):
        self.max_q_delta = 0
        for state in self.q:
            for action in self.q[state]:
                self.max_q_delta = max(self.max_q_delta, self.q_delta[state][action])
        return self.max_q_delta

    def step(self, show_deltas=True):
        self.old_q_table = copy.deepcopy(self.q)
        self.q = self.update_q_table()
        self.q_delta = self.compute_q_delta(self.q, self.old_q_table)
        self.max_q_delta = self.compute_max_q_delta()
        return self.max_q_delta

if __name__ == "__main__":
    example = GridWorldQLearning()
    max_delta = 1
    num_iterations = 0
    min_num_iterations = 20
    max_delta_history = []
    while ( (max_delta > .005) and (num_iterations < min_num_iterations)):
        max_delta = example.step()
        max_delta_history.append(max_delta)
    print(max_delta_history, len(max_delta_history), sep="\n\n")

