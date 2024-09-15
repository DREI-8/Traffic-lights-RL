# Traffic-lights-RL

Notes:
When creating or editing the network, ensure that the duration of each traffic light phase is set to a value greater than the total simulation time (which is specified when initializing the environment using the env class, with a default of 1000). This ensures that the traffic lights only switch when explicitly triggered by the step function, preventing unintended phase changes during the simulation.
