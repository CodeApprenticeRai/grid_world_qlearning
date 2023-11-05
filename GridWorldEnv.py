'''
Class conforms to the OpenAI
gym API when not complicated.
'''
import gymnasium as gym
from GridSpace import GridSpace 

class GridWorldEnv:
    def __init__(self, num_rows=4, 
                num_cols=4, 
                inaccessible=[(1,1)],
                success=[(0,3)],
                fail=[(1,3)]):
        
        # 0-up, 1-right, 2-down, 3-left
        # Clockwise increments start from up
        self.action_space = gym.spaces.Discrete(4)
        self.action_space_labels = ["up", "right",
            "down", "left" 
        ]
        
        # self.observation_space
        self.observation_space = GridSpace(
            num_rows,
            num_cols,
            inaccessible,
            success,
            fail
        )
        self.num_rows = num_rows
        self._num_cols = num_rows
        


        # inaccessible states are a member
        # of this set
        self.inaccessible = set(inaccessible)
        
        self.standard_reward = -0.08

        self.success = set(success)
        self.success_reward = 1

        self.fail = set(fail)
        self.fail_reward = -1

        self.curr = (0,0)
        self.terminated = False
    
    def reset(self):
        self.curr = (0,0)
        self.terminated = False
        return (self.curr, {})
    
    def _transfer(self, action):
        next_state = self.curr
        match action:
            case 0: # case up
                next_state = (max(0,self.curr[0]-1), self.curr[1])
            case 1: # case right
                next_state = (self.curr[0], min(self.curr[1]+1, self.observation_space.num_cols-1))
            case 2: # case down
                next_state = (min(self.curr[0]+1, self.observation_space.num_rows-1), self.curr[1])
            case 3: # case left
                next_state = (self.curr[0], max(0, self.curr[1]-1))
        if next_state in self.inaccessible:
            return self.curr
        self.curr = next_state
        return self.curr

    def _reward(self, state):
        if state in self.success:
            return self.success_reward
        elif state in self.fail:
            return self.fail_reward
        else:
            return self.standard_reward

    def step(self, action):
        # Don't call step if the episode is over
        assert (self.terminated == False)
        assert (self.action_space.contains(action))

        observation = self._transfer(action)
        reward = self._reward(observation)

        self.terminated = ((observation in self.success) or (observation in self.fail))

        return (observation, reward, self.terminated, False, {})
    
    def render(self):
        # Render to plt heatmap of state values
        raise NotImplementedError
