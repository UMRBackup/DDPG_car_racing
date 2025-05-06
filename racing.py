import gymnasium as gym
from DDPG import Agent
import torch
import time

def demonstrate_agent(model_path, episodes=5):
    # create environment
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.8)
    
    # create agent
    agent = Agent()
    try:
        agent.load_model(model_path)
        print(f"Model loading succeeded: {model_path}")
    except Exception as e:
        print(f"Loading failed: {e}")
        return
    
    # evaluation mode
    agent.actor.eval()
    agent.critic.eval()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # update
            state = next_state
            total_reward += reward
            steps += 1
            
            # sleep for observation
            time.sleep(0.01)
        
        print(f"episode {episode + 1}: total step = {steps}, total reward = {total_reward:.2f}")
    
    env.close()

    # clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = "./training_model/model_final"
    
    try:
        demonstrate_agent(MODEL_PATH)
    except KeyboardInterrupt:
        print("\nKeyboardInterruption")