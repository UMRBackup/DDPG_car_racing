import gymnasium as gym
import pygame
import numpy as np
import time

def control_car():
    # Initialize pygame for controller support
    pygame.init()
    pygame.joystick.init()
    
    # Check for connected controllers
    if pygame.joystick.get_count() == 0:
        print("No controller detected! Falling back to keyboard controls...")
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Controller connected: {joystick.get_name()}")
    
    # Create environment
    env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.8)
    
    # Initialize state
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    # Control instructions
    print("\nControl Instructions:")
    print("Keyboard:")
    print("↑: Accelerate")
    print("↓: Brake/Reverse")
    print("←: Turn Left")
    print("→: Turn Right")
    print("Q: Quit Game")
    print("\nController:")
    print("Left Analog Stick: Steering")
    print("RT: Accelerate")
    print("LT: Brake")
    print("Start: Quit Game")
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Initialize action [steering, gas, brake]
        action = np.array([0.0, 0.0, 0.0])
        
        # ...existing code...
        if pygame.joystick.get_count() > 0:
            # Controller inputs
            # Steering with left analog stick
            stick = joystick.get_axis(0)  # Left/Right on left stick
            DEAD_ZONE = 0.2

            if abs(stick) < DEAD_ZONE:
                action[0] = 0.0
            else:
                action[0] = np.sign(stick) * (abs(stick) - DEAD_ZONE) / (1 - DEAD_ZONE)

            # Gas (RT) - Axis 5
            gas = joystick.get_axis(5)
            action[1] = gas
                    
            # Brake (LT) - Axis 4
            brake = joystick.get_axis(4)
            action[2] = brake
                    
            # Quit with Start button
            if joystick.get_button(7):
                done = True

        else:
            # Keyboard controls as fallback
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = -1.0
            elif keys[pygame.K_RIGHT]:
                action[0] = 1.0
            if keys[pygame.K_UP]:
                action[1] = 1.0
            if keys[pygame.K_DOWN]:
                action[2] = 1.0
            if keys[pygame.K_q]:
                done = True
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update state
        state = next_state
        total_reward += reward
        steps += 1
        
        time.sleep(0.01)
    
    env.close()
    pygame.quit()
    print(f"\nGame Over!")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    try:
        control_car()
    except KeyboardInterrupt:
        print("\nKeyboard Interruption")
        pygame.quit()