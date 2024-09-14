import gymnasium as gym
from DQNAgent import DQNAgent
import numpy as np
import cv2
import numpy as np
import time


def preprocess_frame(frame, new_size=(96, 96)):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize the frame
    resized_frame = cv2.resize(gray, new_size)
    
    # Normalize pixel values (optional)
    normalized_frame = resized_frame / 255.0
    
    return normalized_frame

def stack_frames(stacked_frames, new_frame, is_new_episode, stack_size):
    # Preprocess the new frame  
    size = (96,96)
    frame = preprocess_frame(new_frame, new_size=size)
    
    if is_new_episode:
        # When new episode is starting, we stack the first frame 4 times as a start, and after we keep adding frames to the stack. 
        stacked_frames = np.stack([frame] * stack_size, axis=0)
    else:
        # Append the new frame to the stack (and remove the oldest frame)
        stacked_frames = np.append(stacked_frames[1:, :, :], np.expand_dims(frame, 0), axis=0)
    
    return stacked_frames


# Create the environment
env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human", continuous=False)

#############################################################################################################################################################################################
# SETUP
#############################################################################################################################################################################################

# i guess we reduce the shape to 96,96 as input? We input 4 frames everytime of size 96,96 so thats why we have this input.  
state_shape = (4,96,96)
agent = DQNAgent(env.action_space, state_shape= state_shape)

# we are going to stack 4 frames on top of each other as input.
stacked_frames = np.zeros(state_shape)

# Timing variables
start_time = time.time()
frame_count = 0



# Run for a certain number of steps
num_episodes = 1000
for _ in range(num_episodes):
    observation, info = env.reset()
    stacked_frames = stack_frames(stacked_frames, observation, is_new_episode = True, stack_size = 4)
    total_reward = 0
    done = False

    while not done:
        frame_count += 1

        action = agent.pick_action(stacked_frames)
        observation, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
        # print(reward)

        total_reward += reward

        # Preprocess and put the observation (frame) we just did in the back of the stack. 
        next_stacked_frames = stack_frames(stacked_frames, observation, is_new_episode=False, stack_size=4)
        agent.remember(stacked_frames, action, reward, next_stacked_frames, terminated)

        # update the DQN network & weights by sampling memory and comparing target Q values with Q values to backprogagate.
        agent.replay()
        agent.update_epsilon()
        stacked_frames = next_stacked_frames
        
        done = truncated or terminated

        # FPS calculation
        if frame_count % 100 == 0:  # Update FPS every 100 frames
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

env.close()















