import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
import random
import copy


SAVE = "./training_model_2/model"  # model save path
LOAD = None  # model load path, set None for no load


# PID controller for pretraining
class PIDController:
    def __init__(self, kp, ki, kd): # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0 
        
    def compute(self, error, dt=1.0):
        # integral
        self.integral += error * dt
        
        # derivative
        derivative = (error - self.previous_error) / dt
        
        # PID output = kp * error + ki * integral + kd * derivative
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # update previous error
        self.previous_error = error
        
        return output

    def reset(self):
        self.previous_error = 0
        self.integral = 0

# DDPG actor network
class Actor(nn.Module):
    def __init__(self, hidden_size=512, sequence_length=4):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # CNN layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=6272,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.lstm.flatten_parameters()

        # action output layer
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
            nn.Tanh()
        )
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # data preprocessing
        if len(x.size()) == 5:  # [batch, sequence, channel, height, width]
            sequence_length = x.size(1)
            x = x.view(-1, *x.size()[2:])  # flatten the sequence dimension
        else:
            sequence_length = 1
            
        # CNN
        x = x / 255.0
        features = self.conv_layers(x)
        features = features.view(batch_size, sequence_length, -1)
        
        # LSTM
        self.lstm.flatten_parameters()
        if hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, hidden)
        
        # only take the last output of LSTM
        lstm_out = lstm_out[:, -1]
        
        # action output
        actions = self.fc_layers(lstm_out)
        return actions, hidden

# DDPG critic network
class Critic(nn.Module):
    def __init__(self, hidden_size=512, sequence_length=4):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # CNN layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=6272,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.lstm.flatten_parameters()

        self.action_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )

        # Q value output layer
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size + 64, 512),  # hidden_size + action_fc output size
            nn.LayerNorm(512),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, state, action, hidden=None):
        batch_size = state.size(0)
        
        # data preprocessing
        if len(state.size()) == 5:
            sequence_length = state.size(1)
            state = state.view(-1, *state.size()[2:])
        else:
            sequence_length = 1
            
        # CNN features extraction
        state = state / 255.0
        features = self.conv_layers(state)
        features = features.view(batch_size, sequence_length, -1)
        
        # LSTM
        self.lstm.flatten_parameters()
        if hidden is None:
            lstm_out, hidden = self.lstm(features)
        else:
            lstm_out, hidden = self.lstm(features, hidden)
            
        lstm_out = lstm_out[:, -1]

        action_features = self.action_fc(action)
        
        # combine state and action features
        x = torch.cat([lstm_out, action_features], dim=1)
        
        # Q value output
        q_value = self.fc_layers(x)
        return q_value, hidden

# DDPG agent
class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # create Actor network
        self.actor = Actor().to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # create Critic network
        self.critic = Critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.sequence_length = 10
        self.state_sequence = deque(maxlen=self.sequence_length)
        self.hidden = None
        
        self.memory = ReplayBuffer(200000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        
        # add noise for exploration
        self.noise_std = 0.3
        self.noise_decay = 0.995
        self.min_noise = 0.05
    
    def select_action(self, state):
        # preprocess state
        state = self.preprocess_state(state)
        self.state_sequence.append(state)
        
        # return random action if sequence is not full
        if len(self.state_sequence) < self.sequence_length:
            return np.array([
                random.uniform(-0.2, 0.2),
                random.uniform(0.5, 0.8),
                random.uniform(0, 0.3)
            ])
        
        with torch.no_grad():
            # convert to tensor
            states = torch.stack(list(self.state_sequence)).unsqueeze(0)
            action, self.hidden = self.actor(states, self.hidden)
            action = action.cpu().numpy()[0]
        
        # add noise for exploration
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        return action

    def preprocess_state(self, state):
        # resize
        state = cv2.resize(state, (84, 84))
        # CHW (Channel, Height, Width)
        state = np.transpose(state, (2, 0, 1))
        state = torch.FloatTensor(state).to(self.device)
        # normaliza to [0, 1]
        state = state / 255.0
        return state

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            
            'noise_std': self.noise_std,
            'hidden': self.hidden,
            'state_sequence': list(self.state_sequence)
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        if not torch.cuda.is_available():
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(path)
        
        # load Actor network
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        
        # load Critic network
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # load other parameters
        self.noise_std = checkpoint['noise_std']
        self.hidden = checkpoint['hidden']
        
        # restore state sequence
        self.state_sequence.clear()
        for state in checkpoint['state_sequence']:
            self.state_sequence.append(state)
        
        print(f"Model loaded from {path}")

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # sample a batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # preprocess states and next_states
        states = torch.stack([self.preprocess_state(s) for s in states])
        next_states = torch.stack([self.preprocess_state(s) for s in next_states])
        
        # convert to tensors and move to device
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        self.actor.train()
        self.critic.train()

        # training Critic network
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)
            next_q_values, _ = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_values, _ = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # training Actor network
        actions_pred, _ = self.actor(states)
        actor_loss = -self.critic(states, actions_pred)[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        # update exploration noise
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

    def select_action(self, state):
        # preprocess state
        state = self.preprocess_state(state)
        self.state_sequence.append(state)
        
        # return random action if sequence is not full
        if len(self.state_sequence) < self.sequence_length:
            return np.array([
                random.uniform(-1, 1),
                random.uniform(0, 1),
                random.uniform(0, 1)
            ], dtype=np.float32)
        
        with torch.no_grad():
            # set the actor to evaluation mode
            self.actor.eval()
            states = torch.stack(list(self.state_sequence)).unsqueeze(0)
            action, self.hidden = self.actor(states, self.hidden)
            action = action.cpu().numpy()[0]
            self.actor.train()

        # ensure action is in the correct range
        if len(action) != 3:
            print(f"Warning! Action space length error: {len(action)}")
            action = np.zeros(3, dtype=np.float32)        
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action.astype(np.float32)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        action = np.array(action, dtype=np.float32).flatten()
        assert len(action) == 3, f"Warning! Action space length error: {len(action)}"
        
        # prioritize the experience based on the reward
        priority = abs(reward) + 1.0
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        # get indices based on priorities
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def get_track_features(obs):
    # Preprocess the observation
    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # detect track and green areas
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 30, 180])
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    track_mask = cv2.inRange(blurred, lower_gray, upper_gray)
    green_mask = cv2.inRange(blurred, lower_green, upper_green)
    combined_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(green_mask))
    
    # ROI area
    height = track_mask.shape[0]
    roi_height = int(height * 0.2)
    roi_bottom = int(height * 0.9)
    roi = combined_mask[roi_height:roi_bottom, :]
    
    # seperate near and far ROI
    near_roi = roi[int(roi.shape[0]*0.4):, :]
    far_roi = roi[:int(roi.shape[0]*0.6), :]
    
    return near_roi, far_roi, roi.shape[1]

# controller for pretraining
class PretrainController:
    def __init__(self):
        self.steering_pid = PIDController(kp=0.2, ki=0.0001, kd=0.3)
        self.speed_pid = PIDController(kp=0.15, ki=0.0001, kd=0.3)
        self.prev_steer = 0.0
        self.prev_brake = 0.0
        
    def get_action(self, obs):
        near_roi, far_roi, width = get_track_features(obs)
        
        # detect lane center
        contours, _ = cv2.findContours(near_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x = width // 2
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
        
        # calculate steering
        center_diff = center_x - width/2
        dead_zone = width * 0.02
        if abs(center_diff) < dead_zone:
            center_diff = 0
            
        raw_steer = self.steering_pid.compute(center_diff/(width/3))
        raw_steer = np.clip(raw_steer, -1, 1)
        
        # smooth steering
        smooth_factor = 0.85
        steer = raw_steer * (1 - smooth_factor) + self.prev_steer * smooth_factor
        steer = np.clip(steer, -1, 1)
        
        # base gas and brake
        gas = 0.4
        brake = 0.0
        
        if abs(steer) > 0.5:  # slow down on sharp turns
            gas *= (1 - abs(steer) * 0.5)
            brake = abs(steer) * 0.2
            
        self.prev_steer = steer
        
        return np.array([steer, gas, brake], dtype=np.float32)

# pretraining part
def pretrain_agent(agent, env, episodes):
    print("Pretraining start...")
    controller = PretrainController()
    FRAME_SKIP = 3
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        frame_count = 0
        last_action = None


        while not done:
            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                # get action from PID controller
                action = controller.get_action(state) 
                # add noise for exploration
                noise = np.random.normal(0, 0.1, size=action.shape)
                action = np.clip(action + noise, -1, 1)
                last_action = action
            else:
                action = last_action
            
            # actions
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if frame_count % FRAME_SKIP == 0:
                agent.memory.push(state, action, reward * FRAME_SKIP, next_state, done)
                steps += 1
                if len(agent.memory) > agent.batch_size:
                    agent.train()

            state = next_state
            total_reward += reward
        
        print(f"Pretraining episode: {episode+1}/{episodes}, step: {steps}, reward: {total_reward:.2f}")
    
    print("Pretraining complete.")

# training part
def train_agent(render, load_path=LOAD, save_path=SAVE):

    render_mode = "human" if render else "rgb_array"
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    agent = Agent()
    PRETRAIN_EPISODES = 40
    PRETRAIN = True
    FRAME_SKIP = 3

    # Pretrain
    if not load_path and PRETRAIN:
        print("No model loaded. Training with pretraining...")
        pretrain_agent(agent, env, PRETRAIN_EPISODES)
    elif not load_path and not PRETRAIN:
        print("Training without loading and pretraining...") 
    else:
        try:
            agent.load_model(load_path)
            print(f"Model loading succeeded: {load_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")

    episodes = 5000 # takes about 150GB memory for 5000 episodes
    best_reward = -float('inf') # -inf for initialization to make sure the first reward is better than this value

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            frame_count = 0
            last_action = None
            
            while not done:
                frame_count += 1

                if frame_count % FRAME_SKIP == 0:
                    # action selection
                    action = agent.select_action(state)
                    last_action = action
                else:
                    action = last_action

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if frame_count % FRAME_SKIP == 0:
                    agent.memory.push(state, action, reward * FRAME_SKIP, next_state, done)
                    agent.train()
                
                state = next_state
                total_reward += reward
            
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

            # checkpoint saving every 10 episodes
            if episode % 10 == 0:
                checkpoint_path = f"{save_path}_episode_{episode}"
                try:
                    agent.save_model(checkpoint_path)
                except Exception as e:
                    print(f"Checkpoint saving failed: {e}")
            
            # save the best model
            if total_reward > best_reward:
                best_reward = total_reward
                try:
                    agent.save_model(f"{save_path}_best")
                except Exception as e:
                    print(f"Best model saving failed: {e}")
            
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")

    finally:
        # save the final model
        try:
            final_path = f"{save_path}_final"
            agent.save_model(final_path)
            print(f"Final model saved to: {final_path}")
        except Exception as e:
            print(f"Model saving failed: {e}")
        
        print(f"Training complete. Best reward: {best_reward:.2f}")
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train_agent(render=False) # set render=True to visualize the training