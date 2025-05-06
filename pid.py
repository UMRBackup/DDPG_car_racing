import gymnasium as gym
import numpy as np
from collections import defaultdict
import cv2

class CarRacing:
    def __init__(
            self,
            env: gym.Env,
            direction: str = 'CCW',
            use_random_direction: bool = True,
            backwards_flag: bool = True,
    ):
        self.env = env
        self.direction = direction
        self.use_random_direction = use_random_direction
        self.backwards_flag = backwards_flag
        self.h_ratio = 0.25
        self.use_ego_color = False

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.previous_error = 0 
        self.integral = 0 
        
    def compute(self, error, dt=1.0):

        self.integral += error * dt

        derivative = (error - self.previous_error) / dt
        
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # update previous error for next iteration
        self.previous_error = error
        
        return output

    def reset(self):
        self.previous_error = 0
        self.integral = 0

def my_policy(obs):
    # initialize action variables
    gas = 0.0
    brake = 0.0
    steer = 0.0

    # initialize PID controllers
    if not hasattr(my_policy, 'steering_pid'):
        my_policy.steering_pid = PIDController(kp=0.2, ki=0.0001, kd=0.4)
    if not hasattr(my_policy, 'speed_pid'):
        my_policy.speed_pid = PIDController(kp=0.15, ki=0.0001, kd=0.3)

    if not hasattr(my_policy, 'debug_counter'):
        my_policy.debug_counter = 0

    # initialize history
    if not hasattr(my_policy, 'prev_steer'):
        my_policy.prev_steer = 0.0
    if not hasattr(my_policy, 'straight_count'):
        my_policy.straight_count = 0
    if not hasattr(my_policy, 'start_steps'):
        my_policy.start_steps = 0
    
    # set initial action
    if my_policy.start_steps < 5:
        gas = 0.6
        brake = 0.0
        my_policy.start_steps += 1
        return np.array([0.0, gas, brake], dtype=np.float32)


    if not hasattr(my_policy, 'prev_frame'):
        my_policy.prev_frame = obs.copy()  # initialize with the first frame
        my_policy.estimated_speed = 0.3    # set a default speed
        my_policy.speed_history = [0.3] * 5  # initialize speed history
        return np.array([0.0, 0.4, 0.0], dtype=np.float32) 
    
    # calculate speed using optical flow
    if my_policy.prev_frame is not None:
        try:
            # use green area for calculation
            hsv_curr = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
            hsv_prev = cv2.cvtColor(my_policy.prev_frame, cv2.COLOR_RGB2HSV)
            
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            
            green_curr = cv2.inRange(hsv_curr, lower_green, upper_green)
            green_prev = cv2.inRange(hsv_prev, lower_green, upper_green)
            
            # calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                green_prev, green_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            
            h, w = flow.shape[:2]
            left_flow = flow[:, :w//3]  # left side area
            right_flow = flow[:, 2*w//3:]  # right side area
            
            # left and right speed calculation
            left_speed = np.mean(np.sqrt(left_flow[..., 0]**2 + left_flow[..., 1]**2))
            right_speed = np.mean(np.sqrt(right_flow[..., 0]**2 + right_flow[..., 1]**2))
            
            # calculate average speed
            speed_magnitude = (left_speed + right_speed) / 2
            current_speed = speed_magnitude * 3.0  # scale factor
            
            # smooth speed estimation
            my_policy.speed_history.append(current_speed)
            if len(my_policy.speed_history) > 5:
                my_policy.speed_history.pop(0)
            my_policy.estimated_speed = np.mean(my_policy.speed_history)
            
            # speed correction
            my_policy.estimated_speed = np.clip(my_policy.estimated_speed, 0.1, 1.0)
            
        except Exception as e:
            print(f"Warining! Speed error: {e}")
            my_policy.estimated_speed = np.mean(my_policy.speed_history) if my_policy.speed_history else 0.3

    # save the current frame
    my_policy.prev_frame = obs.copy()

    hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    

    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 30, 180])
    

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # create masks for track and green areas
    track_mask = cv2.inRange(blurred, lower_gray, upper_gray)
    green_mask = cv2.inRange(blurred, lower_green, upper_green)

    # combine masks to exclude green areas from track mask
    combined_mask = cv2.bitwise_and(track_mask, cv2.bitwise_not(green_mask))
    
    # enforce morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    track_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # ROI setting
    height = track_mask.shape[0]
    roi_height = int(height * 0.2)
    roi_bottom = int(height * 0.9)
    roi_width = track_mask.shape[1]
    roi = track_mask[roi_height:roi_bottom, :]

    near_roi = roi[int(roi.shape[0]*0.4):, :] 
    far_roi = roi[:int(roi.shape[0]*0.6), :]  

    # curve detection
    is_curve_ahead = False
    is_curve = False
    curve_direction = 0

    # far curve detection
    far_contours, _ = cv2.findContours(far_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(far_contours) > 0:
        valid_contours = []
        for cnt in far_contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                valid_contours.append(cnt)
        
        if valid_contours:
            far_track_contour = max(valid_contours, key=cv2.contourArea)
            far_rect = cv2.minAreaRect(far_track_contour)
            far_angle = abs(far_rect[2])
            is_curve_ahead = far_angle > 3 and far_angle < 87
            
            # calculate curve severity
            curve_severity = min(abs(far_angle) / 40.0, 1.0)
            
            if is_curve_ahead:
                M = cv2.moments(far_track_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    curve_direction = 1 if cx > far_roi.shape[1]/2 else -1

    # near curve detection
    near_contours, _ = cv2.findContours(near_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x = roi.shape[1] // 2  # follow the center line
    if len(near_contours) > 0:
        valid_near_contours = []
        for cnt in near_contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                valid_near_contours.append(cnt)
        
        if valid_near_contours:
            near_track_contour = max(valid_near_contours, key=cv2.contourArea)
            M = cv2.moments(near_track_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])  # calculate center line
    
    center_diff = center_x - roi.shape[1]/2

    # dead zone for steering
    dead_zone = roi.shape[1] * 0.02
    if abs(center_diff) < dead_zone:
        center_diff = 0

    # steering PID control
    raw_steer = my_policy.steering_pid.compute(center_diff/(roi.shape[1]/3))
    raw_steer = np.clip(raw_steer, -1, 1)

    # smooth steering output
    smooth_factor = 0.85
    steer = raw_steer * (1 - smooth_factor) + my_policy.prev_steer * smooth_factor
    steer = np.clip(steer, -1, 1)

    # speed control
    target_speed = 0.5 
    current_speed = gas 
    
    # adjust target speed based on curve severity
    if is_curve_ahead:
        speed_reduction = 0.8 * curve_severity
        target_speed *= (0.4 - speed_reduction)
        if curve_severity > 0.7:
            target_speed *= 0.5
    elif is_curve:
        target_speed *= 0.2
    
    steering_factor = 1.0 - abs(steer) * 0.8
    target_speed *= steering_factor
    
    # calculate speed error and PID output
    speed_error = target_speed - current_speed
    pid_output = my_policy.speed_pid.compute(speed_error)
    
    gas = np.clip(pid_output, 0.1, 0.8)
    
    # brake logic
    brake = 0.0
    if is_curve_ahead:
        if curve_severity > 0.7:
            brake = 0.4 + curve_severity * 0.4
        elif curve_severity > 0.4:
            brake = 0.2 + curve_severity * 0.2
        else:
            brake = 0.1 + curve_severity * 0.2
 
    if is_curve:
        if curve_severity > 0.7:
            brake = 0.3 + curve_severity * 0.4
        else:
            brake = 0.2 + curve_severity * 0.3 

    if brake > 0.4:
        gas = 0.1 
    elif brake > 0:
        gas *= (1 - brake * 0.5) 

    # ensure minimum power
    min_power = 0.1
    if not (is_curve or is_curve_ahead):
        min_power = 0.2
    elif curve_severity < 0.5:
        min_power = 0.15

    if brake < 0.4: 
        gas = max(gas, min_power)
    # smooth brake
    if not hasattr(my_policy, 'prev_brake'):
        my_policy.prev_brake = 0.0
    brake_smooth = 0.7
    brake = brake * (1 - brake_smooth) + my_policy.prev_brake * brake_smooth
    my_policy.prev_brake = brake

    if brake > 0:
        gas = max(gas, 0.1)  # keep gas

    # update previous info
    my_policy.prev_steer = steer

    if my_policy.debug_counter % 30 == 0:
        if hasattr(my_policy, 'curve_severity'):
            print(f"curve severity: {curve_severity:.2f}, steer: {raw_steer:.2f}, gas: {gas:.2f}, brake: {brake:.2f}")
        else:
            print(f"steer: {raw_steer:.2f}, gas: {gas:.2f}, brake: {brake:.2f}")
    
    my_policy.debug_counter += 1

    return np.array([steer, gas, brake], dtype=np.float32)

def main():
    env_1 = gym.make("CarRacing-v3", render_mode="human", max_episode_steps=1000, lap_complete_percent=0.95)
    env = CarRacing(env_1)
    obs, info = env.env.reset()
    done = False
    total_reward = 0
    while not done:
        action = my_policy(obs)
        obs, reward, terminated, truncated, info = env.env.step(action)
        done = terminated or truncated
        total_reward += reward
    print("individual scores:", total_reward)
    env.env.close()


if __name__ == "__main__":
    main()