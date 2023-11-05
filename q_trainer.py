import gymnasium as gym
import pickle
from GridSpace import GridSpace
from GridWorldEnv import GridWorldEnv
from Q import Q
import numpy as np
import win32api
import matplotlib.pyplot as plt

class q_trainer:
    def __init__(self, state_data_filename, PARAM_number_of_episodes=5, 
                 PARAM_episode_max_length=1000, PARAM_learning_rate=0.999, 
                 PARAM_discount_factor=0.89, PARAM_epsilon=0.2
                 ): 
        self.env = GridWorldEnv() 
        self.state_data_filename = state_data_filename
        
        self.PARAM_number_of_episodes = PARAM_number_of_episodes
        self.PARAM_episode_max_length = PARAM_episode_max_length
        self.learning_rate = PARAM_learning_rate
        self.discount_factor = PARAM_discount_factor
        self.epsilon = PARAM_epsilon
        
        self.state_space_repr = GridSpace()

        self.q = Q(
            state_space_repr=self.state_space_repr,
            action_space_repr=self.env.action_space,
            learning_rate=self.learning_rate, 
            discount_factor=self.discount_factor,
            epsilon=self.epsilon,
            # q_init_value=0
        )


        # Try to restore state of a previous seission
        try:
            with open(state_data_filename, 'rb') as f:
                table = pickle.load(f)
                self.q.q_table = table
        except FileNotFoundError:
            pass

        

        self.mean_reward_per_session = []
        return

    def save_q_table(self):
        with open(self.state_data_filename, 'wb') as f:
            pickle.dump(self.q.q_table, f)
        return
    
    def run_session(self):
        reward_history = []

        for episode in range(self.PARAM_number_of_episodes):
            observation, _ = self.env.reset()

            episode_reward = 0
            misses_at_start = self.q.state_lookup_misses
            for t in range(self.PARAM_episode_max_length):
                # if (self.env.render_mode == "human"):
                # self.env.render()
                action = self.q.best_action(observation)
                next_observation, reward, terminate_episode_signal, _, step_info = self.env.step(action)
                delta = self.q.update(observation, action, reward, next_observation)
                
                # state_watch = (observation, self.env.action_space_labels[action], reward, next_observation, delta)
                # print(state_watch)
                

                episode_reward += reward
                observation = next_observation

                if terminate_episode_signal:
                    break
                

            
            reward_history.append(episode_reward)

            misses_during_episode = self.q.state_lookup_misses - misses_at_start
            episodes_left = self.PARAM_number_of_episodes - episode - 1
            exp_info = (episodes_left, episode_reward, t) 
            print(exp_info)
            self.heatmap()
        
        session_reward = np.mean(reward_history)
        print((session_reward))
        self.mean_reward_per_session.append(session_reward)

        self.save_q_table()
        win32api.MessageBox(0, 'Session Done', 'DoneMessage', 0x00001000)
        return

    def heatmap(self):
        self.for_heatmap = []
        for i in range(self.state_space_repr.num_rows):
            row = []
            for j in range(self.state_space_repr.num_cols):
                state = self.q.q_table[(i,j)]
                avg_value = np.mean(list(state.values()))
                row.append(avg_value)
            self.for_heatmap.append(row)
        fig, ax = plt.subplots()
        im = ax.imshow(self.for_heatmap)

        for i in range(self.state_space_repr.num_rows):
            for j in range(self.state_space_repr.num_cols):
                text_to_show = "{:0.5f}".format(self.for_heatmap[i][j])
                ax.text(j, i, text_to_show,
                    ha='center', va='center', color='w',
                    
                )

        plt.show()
        return