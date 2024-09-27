import gymnasium as gym
from DQNAgent import DQNAgent
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def preprocess_frame(frame, new_size=(96, 96)):
    """Convert the frame to grayscale, resize, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray, new_size)
    normalized_frame = resized_frame / 255.0
    return normalized_frame


def stack_frames(stacked_frames, new_frame, is_new_episode, stack_size):
    """Stack frames for input to the DQN model."""
    frame = preprocess_frame(new_frame, new_size=(96, 96))
    if is_new_episode:
        stacked_frames = np.stack([frame] * stack_size, axis=0)
    else:
        stacked_frames = np.append(stacked_frames[1:, :, :], np.expand_dims(frame, 0), axis=0)
    return stacked_frames

def printGraph(rewards_from_episodes):
    """Plot the rewards from episodes"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_from_episodes, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Configuration parameters
    state_shape = (4, 64, 64)  # Stack 4 frames of size 96x96
    max_frames_per_episode = 2000
    num_episodes = 751
    skip_frames = 2
    epsilon = 0
    epsilon_min = 0.1
    epsilon_decay = 0.999
    gamma = 0.95
    batch_size = 64
    memory_capacity = 15000
    learning_rate = 0.0005
    target_update_frequency = 10
    model_directory = "models"
    model_filename = "dqn_model"
    negative_reward_limit = 30

    # Initialize environment and agent
    env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="human", continuous=False)
    
    # Pass parameters to DQNAgent
    agent = DQNAgent(
        action_space=env.action_space,
        state_shape=state_shape,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        batch_size=batch_size,
        memory_capacity=memory_capacity,
        learning_rate=learning_rate,
        target_update_frequency=target_update_frequency,
        model_directory=model_directory,
        model_filename=model_filename
    )

    # Initialize frame stacking
    stacked_frames = np.zeros(state_shape)

    # Load saved model (if any)
    start_episode, rewards_from_episodes  = agent.load_model()
    

    # rewards_from_episodes = []
    # Run episodes
    for episode in range(start_episode, num_episodes):
        # Reset environment and frames
        observation, info = env.reset()
        stacked_frames = stack_frames(stacked_frames, observation, is_new_episode=True, stack_size=state_shape[0])
        done = False
        frame_in_episode = 0
        total_reward_episode = 0
        negative_reward_count = 0

        # Cumulative reward for the episode
        

        # Episode loop
        while not done and frame_in_episode < max_frames_per_episode:
            frame_start_time = time.time()

            # Select action
            if frame_in_episode % skip_frames == 0:
                action = agent.pick_action(stacked_frames)

            # Initialize frame reward accumulation
            total_step_reward = 0

            # Step through the environment, skipping frames
            for _ in range(skip_frames):
                observation, reward, terminated, truncated, info = env.step(action)
                total_step_reward += reward  # Accumulate reward over the skipped frames
                if total_step_reward > 2.5:
                    total_step_reward = 4
                if action == 3:
                    total_step_reward += 0.5
                if terminated or truncated:
                    done = True
                    break
                

            # Update the agent after processing skip_frames
            # if frame_in_episode % skip_frames == 0:
            #     next_stacked_frames = stack_frames(stacked_frames, observation, is_new_episode=False, stack_size=state_shape[0])
            #     agent.remember(stacked_frames, action, total_step_reward, next_stacked_frames, terminated)
            #     agent.replay()
            #     agent.update_epsilon()

            #     # Update stacked frames
            #     stacked_frames = next_stacked_frames
                
                
                
                
            if total_step_reward <= 0.9:
                negative_reward_count += 1
            else:
                negative_reward_count = 0
                    
            if frame_in_episode > 100 and negative_reward_count >= negative_reward_limit:
                print(f"Ending episode {episode} early due to too many negative rewards.")
                break

            # Check for termination or truncation
            done = terminated or truncated
            frame_in_episode += 1

            # Timing report
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            print(f"Episode {episode}. Frame {frame_in_episode} took {frame_time:.4f} seconds with cumulative reward: {total_step_reward:.4f}")
            total_reward_episode += total_step_reward
            
        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()
            

        # Save model every 50 episodes
        # if episode % 50 == 0:
        #     agent.save_model(episode, rewards_from_episodes)
        
        rewards_from_episodes.append(total_reward_episode) 
        print(rewards_from_episodes)
            
    
    # Close environment
    env.close()
    printGraph(rewards_from_episodes)


if __name__ == "__main__":
    main()
