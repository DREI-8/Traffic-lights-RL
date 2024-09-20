import os
import numpy as np
import pytest
from env import SumoEnv

# Number of steps taken to verify that lane densities and queues are increasing, adjust depending on the simulation apparition probabilities
NUM_STEPS_TO_TEST_DENS = 10

# This test file tests the SumoEnv environment for each environment folder in the data directory
environments = [d for d in os.listdir('./data') if os.path.isdir(os.path.join('./data', d))]

@pytest.mark.parametrize("env_path", environments)
def test_sumo_env(env_path):
    """Test the SumoEnv environment for each environment folder"""
    
    env = SumoEnv(env_path)

    observation = env.reset()
    assert isinstance(observation, np.ndarray), "reset() should return a numpy ndarray"
    
    observation_space = env.get_observation_space()
    phase_one_hot_shape = observation_space["phase_one_hot"]

    phase_one_hot = observation[:phase_one_hot_shape[0]]
    assert sum(phase_one_hot) == 1, "There should be exactly one '1' in phase_one_hot and the rest '0'"

    action_space = env.get_action_space()
    assert isinstance(action_space, np.ndarray), "get_action_space() should return a numpy ndarray"

    assert isinstance(observation_space, dict), "get_observation_space() should return a dictionary"
    
    total_shape = observation_space["total_shape"]
    assert observation.shape == total_shape, "The shape of the observation should match the total shape of the observation space"

    new_observation, reward, done, info = env.step(np.zeros_like(action_space))
    assert isinstance(new_observation, np.ndarray), "step() should return a numpy ndarray as the observation"
    assert isinstance(reward, float), "step() should return a float as the reward"
    assert isinstance(done, bool), "step() should return a boolean as 'done'"
    assert isinstance(info, dict), "step() should return a dictionary as 'info'"
    assert new_observation.shape == total_shape, "The shape of the observation after step should match the observation space"

    min_green_idx = phase_one_hot_shape[0]
    assert new_observation[min_green_idx] == 0, "min_green should be 0 after reset"

    env.reset()

    for _ in range(env.MIN_SWITCH):
        observation, reward, done, info = env.step(np.zeros_like(action_space))

    assert observation[min_green_idx] == 1, "min_green should have switched to 1 after MIN_SWITCH steps"

    lane_densities_start = min_green_idx + 1
    lane_queues_start = lane_densities_start + len(observation_space["lane_densities"])
    
    previous_density = np.mean(observation[lane_densities_start:lane_queues_start])
    previous_queue = np.mean(observation[lane_queues_start:])
    
    for _ in range(NUM_STEPS_TO_TEST_DENS):
        observation, reward, done, info = env.step(np.zeros_like(action_space))

    new_density = np.mean(observation[lane_densities_start:lane_queues_start])
    new_queue = np.mean(observation[lane_queues_start:])
    
    assert new_density > previous_density, "Lane densities should increase over time (you can increase the number of steps if this fails, at the top of the test file)"
    assert new_queue > previous_queue, "Lane queues should increase over time, (you can increase the number of steps if this fails, at the top of the test file)"

    observation, reward, done, info = env.step(np.ones_like(action_space))

    assert observation[min_green_idx] == 0, "min_green should be 0 after action 1"
    assert observation[0] == 0 and observation[1] == 1, "phase_one_hot should change from first bit to second after action 1"

    env.reset()
    steps_taken = 0
    while not done:
        observation, reward, done, info = env.step(np.zeros_like(action_space))
        steps_taken += 1

    assert steps_taken == env.sim_duration, "The episode should end after sim_duration steps"
    assert done, "The 'done' flag should be True at the end of the episode"

    env.reset()
    env.close()
