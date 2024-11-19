## This is not a RL agent, but a fixed time agent that is used as a benchmark for the RL agents.

class fixed_time_agent():
    """
    Fixed time agent for controlling traffic lights.

    Methods
    -------
    __init__(env, dur_greens, dur_reds)
        Initializes the fixed time agent with the given environment and phase durations.
    act(observation)
        Determines the action to take based on the current observation.
    play_episode()
        Plays an episode in the environment and returns the average density and queue length.
    """
    def __init__(self, env, dur_greens, dur_reds):
        """
        Initializes the fixed time agent with the specified environment and phase durations.

        Parameters
        ----------
        env : object
            The environment where the agent operates, providing access to state and action spaces.
        dur_greens : int or list
            Duration(s) for the green light phase(s).
        dur_reds : int or list
            Duration(s) for the red light phase(s).
        """
        self.env = env
        self.dur_yellow = env.YEL_DURATION
        self.obs_space = env.get_observation_space()
        self.action_space = env.get_action_space()

        if isinstance(dur_greens, int):
            self.dur_greens = [dur_greens]
        else:
            self.dur_greens = dur_greens
        if isinstance(dur_reds, int):
            self.dur_reds = [dur_reds]
        else:
            self.dur_reds = dur_reds

        self.counter = [1 for _ in range(len(self.dur_greens))]

    def act(self, observation):
        """
        Determines the action to take based on the current state observation.

        Parameters
        ----------
        observation : list
            The current state observation from the environment. See the get_observation method in the environment class.

        Returns
        -------
        tuple
            A tuple containing the new observation, the reward received, and a boolean indicating if the episode is done.

        Raises
        ------
        ValueError
            If the minimum switch time has not been reached, indicating that the duration for the green or red phase
            is shorter than allowed by the environment configuration.
        """
        action = [0 for _ in range(len(self.action_space))]

        for i, tl_info in enumerate(self.obs_space["infos_per_TL"]):
            min_switch = tl_info["min_green"][0]

            if observation[tl_info["phase_one_hot"][0]] == 1 and self.counter[i] < self.dur_greens[i]:
                self.counter[i] += 1
                action[i] = 0

            elif observation[tl_info["phase_one_hot"][0]] == 1 and self.counter[i] == self.dur_greens[i]:
                if observation[min_switch] == 0:
                    raise ValueError("The minimum switch time has not been reached yet. Check that the duration for the green phase choosen is more than the minimum switch duration set in the env.py file.")
                else:
                    self.counter[i] = 0
                    action[i] = 1
            
            elif observation[tl_info["phase_one_hot"][0]+2] == 1 and self.counter[i] < self.dur_reds[i]:
                self.counter[i] += 1
                action[i] = 0
            
            elif observation[tl_info["phase_one_hot"][0]+2] == 1 and self.counter[i] == self.dur_reds[i]:
                if observation[min_switch] == 0:
                    raise ValueError("The minimum switch time has not been reached yet. Check that the duration for the red phase choosen is more than the minimum switch duration set in the env.py file.")
                else:
                    self.counter[i] = 0
                    action[i] = 1
            
            elif self.counter[i] < self.dur_yellow and (observation[tl_info["phase_one_hot"][0]+1] == 1 or observation[tl_info["phase_one_hot"][0]+3] == 1):
                self.counter[i] += 1
                action[i] = 0
            
            elif self.counter[i] == self.dur_yellow and (observation[tl_info["phase_one_hot"][0]+1] == 1 or observation[tl_info["phase_one_hot"][0]+3] == 1):
                self.counter[i] = 0
                action[i] = 0

        observation, reward, done, _ = self.env.step(action)
        return observation, reward, done
       

    def play_episode(self, render=False):
        """
        Plays a full episode in the environment, controlling the traffic lights using a fixed time strategy.

        Parameters
        ----------
        render : bool
            If True, the environment will be rendered. The SUMO GUI will be launched, and you will need to press 
            the "Play" button to start the simulation and close the window manually.

        Returns
        -------
        float
            The total cumulative reward for the episode.
        """

        if render:
            observation = self.env.reset(True)
        else:
            observation = self.env.reset()

        done = False
        total_reward = 0

        while not done:
            observation, reward, done = self.act(observation)
            total_reward += reward

        return total_reward