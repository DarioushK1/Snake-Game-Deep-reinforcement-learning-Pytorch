import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import numpy as np
import game

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class Conv_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define convolutional layers
        self.final_conv = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.final_conv, kernel_size=5, stride=1)

        # Automatically calculate the size of the flattened features after convolution layers
        self._to_linear = None
        # self.convs(torch.randn(3, 50, 40).view(-1, 3, 50, 40))  # Dummy input for size calculation

        # Define fully connected layers
        # print(self._to_linear)
        self.fc1 = nn.Linear(self.final_conv, 256)  # Input size depends on conv layer output
        self.fc2 = nn.Linear(256, 4 if game.ACTION_TYPE else 3)

    def convs(self, x):
        # Apply convolution layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = torch.max(x, (-1))[0]
        x = torch.max(x, (-1))[0]

        # Calculate the flattened size
        # if not self._to_linear:
        #     self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = self.convs(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
    def __init__(self, model, target ,lr, gamma,DDQN = False, device = None):
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target
        self.DDQN = DDQN
        # self.optimizer = optim.RMSprop(model.parameters())
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        # print("state shape:",state.shape)
        if isinstance(state, np.ndarray):
        # This means there is only one state
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state,axis =0)
            action = np.expand_dims(action,axis =0)
            reward = np.expand_dims(reward,axis = 0)
            done = np.expand_dims(done,axis = 0)
            # print(state.shape)
        else:
            # Here, states is a tuple/list of multiple states
            state = np.stack(state)
            next_state = np.stack(next_state)
            action = np.stack(action)
            reward = np.stack(reward)
            done = np.stack(done)

        # print("Processed states shape:", np.shape(state))
        state = torch.from_numpy(state).float()
        state = state.permute(0,3,1,2)  # Change shape from (H, W, C) to (C, H, W)
        next_state = torch.from_numpy(next_state).float()
        next_state = next_state.permute(0,3,1,2)
        # state = torch.tensor(state, dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        # done = done.to(self.device)

        # 1: predicted Q values with current state
        # print(state.shape, next_state.shape)
        pred = self.model(state)

        target = pred.clone()

        if not self.DDQN:
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new
        else:
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = F.softmax(target, dim=1)
        self.optimizer.zero_grad()
        # print(target, pred)
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        if self.DDQN:
            self.target_model.load_state_dict(self.model.state_dict())
