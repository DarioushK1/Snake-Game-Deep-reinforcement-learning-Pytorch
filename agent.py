import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, Conv_QNet
from helper import plot
import copy
import pygame.surfarray as surfarray
import matplotlib.pyplot as plt
import cv2
import game

MAX_MEMORY = 100000
BATCH_SIZE = 100
LR = 0.00001

class Agent:

    def __init__(self,DDQN=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_games = 0
        self.epsilon = 1 # randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet().to(self.device)
        # self.model = Linear_QNet(11, 256, 3)
        self.DDQN = DDQN
        if self.DDQN:
            self.target_model = copy.deepcopy(self.model)
            self.target_model.to(self.device)
            self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma,DDQN = self.DDQN, device = self.device)
        else:
            self.trainer = QTrainer(self.model, None, lr=LR, gamma=self.gamma,DDQN = self.DDQN, device = self.device)


    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)

    #     dir_l = game.direction == Direction.LEFT
    #     dir_r = game.direction == Direction.RIGHT
    #     dir_u = game.direction == Direction.UP
    #     dir_d = game.direction == Direction.DOWN

    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or
    #         (dir_l and game.is_collision(point_l)) or
    #         (dir_u and game.is_collision(point_u)) or
    #         (dir_d and game.is_collision(point_d)),

    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or
    #         (dir_d and game.is_collision(point_l)) or
    #         (dir_l and game.is_collision(point_u)) or
    #         (dir_r and game.is_collision(point_d)),

    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or
    #         (dir_u and game.is_collision(point_l)) or
    #         (dir_r and game.is_collision(point_u)) or
    #         (dir_l and game.is_collision(point_d)),

    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,

    #         # Food location
    #         game.food.x < game.head.x,  # food left
    #         game.food.x > game.head.x,  # food right
    #         game.food.y < game.head.y,  # food up
    #         game.food.y > game.head.y  # food down
    #         ]

    #     return np.array(state, dtype=int)


    def get_state(self,game):
        # Capture the screen as a numpy array
        screen_array = surfarray.array3d(game.display)
        # gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        # resized = cv2.resize(gray,(80,80))
        state = screen_array[::4, ::4, :]
        # plt.figure()
        # plt.imshow(state)
        # plt.axis('off')  # This will remove the x and y axis
        # plt.show()
        # plt.savefig("state.png", dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # print(state.shape)
        # state = np.transpose(screen_array, (2, 0, 1))
        # state = screen_array
        state = state / 255
        return state


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        # print(len(mini_sample))
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0, 0] if game.ACTION_TYPE else [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 3) if game.ACTION_TYPE else random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.from_numpy(state).float().to(self.device)
            state0 = state0.permute(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
            state0 = state0.unsqueeze(0)  # Add a batch dimension if necessary
            # state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    update_target_every = 2  # Update target model every 5 games
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        print(final_move.index(1), end=", ")

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.epsilon = max(agent.epsilon*agent.epsilon_decay,agent.epsilon_min)
            if agent.n_games % update_target_every == 0 and agent.DDQN:
                agent.trainer.update_target_model()
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Reward:', reward)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            print("\n")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()