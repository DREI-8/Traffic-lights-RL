import os
import numpy as np
import pytest
from Sim import SumoEnv

# Number of steps taken to verify that lane densities and queues are increasing, adjust depending on the simulation apparition probabilities
NUM_STEPS_TO_TEST_DENS = 25

# Making sure the test file works in both the root and Sim directory
data_dir = './data' if os.path.exists('./data') else './Sim/data'

# This test file tests the SumoEnv environment for each environment folder in the data directory
environments = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

@pytest.mark.parametrize("env_path", environments)
def test_sumo_env(env_path):
    """Test the SumoEnv environment for each environment folder"""
    
    env = SumoEnv(env_path, 250, 0.3, 0.1, 0)

    observation = env.reset()
    assert isinstance(observation, np.ndarray), "reset() should return a numpy ndarray"
    
    observation_space = env.get_observation_space()
    total_shape = observation_space["total_shape"]
    
    assert observation.shape == total_shape, "The shape of the observation should match the total shape of the observation space"

    assert isinstance(observation_space, dict), "get_observation_space() should return a dictionary"
    assert "infos_per_TL" in observation_space, "'infos_per_TL' should be a key in the observation space"
    assert "total_shape" in observation_space, "'total_shape' should be a key in the observation space"

    action_space = env.get_action_space()
    assert isinstance(action_space, np.ndarray), "get_action_space() should return a numpy ndarray"
    
    # Iterate over each traffic light and check the observation structure
    for i, tl_info in enumerate(observation_space["infos_per_TL"]):
        phase_one_hot_indices = tl_info["phase_one_hot"]
        min_green_indices = tl_info["min_green"]

        phase_one_hot = observation[phase_one_hot_indices[0]:phase_one_hot_indices[1]]
        assert sum(phase_one_hot) == 1, f"There should be exactly one '1' in phase_one_hot for TL {i} and the rest '0'"

        min_green_value = observation[min_green_indices[0]:min_green_indices[1]]
        assert min_green_value == 0, f"min_green should be 0 for TL {i} after reset"

    # Perform one step in the environment and verify the structure again
    new_observation, reward, done, info = env.step(np.zeros_like(action_space))
    assert isinstance(new_observation, np.ndarray), "step() should return a numpy ndarray as the observation"
    assert isinstance(reward, float), "step() should return a float as the reward"
    assert isinstance(done, bool), "step() should return a boolean as 'done'"
    assert isinstance(info, dict), "step() should return a dictionary as 'info'"
    assert new_observation.shape == total_shape, "The shape of the observation after step should match the observation space"

    for i, tl_info in enumerate(observation_space["infos_per_TL"]):
        min_green_value = new_observation[tl_info["min_green"][0]:tl_info["min_green"][1]]
        assert min_green_value == 0, f"min_green should be 0 for TL {i} after one step"

    # Reset the environment and simulate until MIN_SWITCH is reached to check min_green switching
    env.reset()

    for _ in range(env.MIN_SWITCH):
        observation, reward, done, info = env.step(np.zeros_like(action_space))

    for i, tl_info in enumerate(observation_space["infos_per_TL"]):
        min_green_value = observation[tl_info["min_green"][0]:tl_info["min_green"][1]]
        assert min_green_value == 1, f"min_green should switch to 1 for TL {i} after MIN_SWITCH steps"

    # Test lane density and queue increasing over time
    for i, tl_info in enumerate(observation_space["infos_per_TL"]):
        lane_densities_start = tl_info["lane_densities"][0]
        lane_densities_end = tl_info["lane_densities"][1]
        lane_queues_start = tl_info["lane_queues"][0]
        lane_queues_end = tl_info["lane_queues"][1]

        previous_density = np.mean(observation[lane_densities_start:lane_densities_end])
        previous_queue = np.mean(observation[lane_queues_start:lane_queues_end])

        for _ in range(NUM_STEPS_TO_TEST_DENS):
            observation, reward, done, info = env.step(np.zeros_like(action_space))

        new_density = np.mean(observation[lane_densities_start:lane_densities_end])
        new_queue = np.mean(observation[lane_queues_start:lane_queues_end])

        assert new_density > previous_density, f"Lane densities should increase over time for TL {i} (adjust NUM_STEPS_TO_TEST_DENS if this fails)"
        assert new_queue > previous_queue, f"Lane queues should increase over time for TL {i} (adjust NUM_STEPS_TO_TEST_DENS if this fails)"

    observation, reward, done, info = env.step(np.ones_like(action_space))

    for i, tl_info in enumerate(observation_space["infos_per_TL"]):
        min_green_value = observation[tl_info["min_green"][0]:tl_info["min_green"][1]]
        assert min_green_value == 0, f"min_green should be 0 for TL {i} after action 1"
        
        # Check if phase_one_hot changes correctly (first bit to second bit after changing phase)
        phase_one_hot = observation[tl_info["phase_one_hot"][0]:tl_info["phase_one_hot"][1]]
        assert phase_one_hot[0] == 0 and phase_one_hot[1] == 1, f"phase_one_hot should change correctly for TL {i} after action 1"

    env.reset()
    steps_taken = 0
    while not done:
        observation, reward, done, info = env.step(np.zeros_like(action_space))
        steps_taken += 1

    assert steps_taken == env.sim_duration, "The episode should end after sim_duration steps"
    assert done, "The 'done' flag should be True at the end of the episode"

    env.reset()
    env.close()
