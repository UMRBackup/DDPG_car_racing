import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
import random


SAVE = "./training_model"
LOAD = None


class CarRacingNet(nn.Module):
    def __init__(self):
        super(CarRacingNet, self).__init__()
        # CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # DNN
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # [steering, gas, brake]
        )
        
    def forward(self, x):
        x = x / 255.0  # normalize to [0, 1]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return torch.tanh(self.fc_layers(x))  # output limited to[-1,1]

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class CarRacingAgent:
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.policy_net = CarRacingNet().to(self.device)
        self.target_net = CarRacingNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer()
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def preprocess_state(self, state):
        # convert to tensor
        state = cv2.resize(state, (84, 84))
        state = np.transpose(state, (2, 0, 1))  # (C, H, W)
        state = torch.FloatTensor(state).to(self.device)
        return state
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            # random action
            return np.array([
                random.uniform(-1, 1),  # steering
                random.uniform(0, 1),   # gas
                random.uniform(0, 1)    # brake
            ])
        
        with torch.no_grad():
            # select action based on policy network
            state = self.preprocess_state(state)
            state = state.unsqueeze(0)
            action_values = self.policy_net(state)
            return action_values.cpu().numpy()[0]
    
    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path):
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # preprocess states and next_states
        states = torch.stack([self.preprocess_state(s) for s in states])
        next_states = torch.stack([self.preprocess_state(s) for s in next_states])        

        # convert to tensors
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q_values = self.policy_net(states)  # [batch_size, 3]

        # calculate target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]  # [batch_size]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values  # [batch_size]

        # convert actions to indices
        batch_size = states.size(0)
        action_indices = torch.arange(batch_size).to(self.device)
        # get the Q values for the actions taken
        current_q = current_q_values[action_indices]  # [batch_size, 3]
        
        # loss function
        loss = nn.MSELoss()(current_q, target_q_values.unsqueeze(1).expand(-1, 3))  # [batch_size, 3]
        
        # optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # epsilon update
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(render, load_path=LOAD, save_path=SAVE):

    render_mode = "human" if render else "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    agent = CarRacingAgent()
    
    # load model
    if load_path:
        try:
            agent.load_model(load_path)
            print(f"成功加载模型: {load_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
    
    episodes = 1000
    best_reward = -float('inf')
    

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # store transition in memory
                agent.memory.push(state, action, reward, next_state, done)
                
                agent.train()
                
                state = next_state
                total_reward += reward
            
            # save model every 100 episodes
            if episode % 100 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
                
                # save model
                agent.save_model(f"{save_path}_episode_{episode}")
                
            # save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(f"{save_path}_best")

    finally:
        print(f"Training complete. Best reward: {best_reward:.2f}")
        agent.save_model(save_path)
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    


if __name__ == "__main__":
    train_agent(render=True) # Set to False for training without rendering