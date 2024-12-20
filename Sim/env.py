import xml.etree.ElementTree as ET
from collections import defaultdict
import traci
import numpy as np
import os

class SumoEnv():
    def __init__(self, sim_name, sim_duration=500, vehicle_prob=0.05, pedestrian_prob=0.05, prob_variation=0.15, seed=None, use_seed_every_episode=True):
        """
        Initialize the environment.

        Parameters
        ----------
        sim_name : str
            The name of the simulation. All related simulation files (.net.xml, .rou.xml, and .sumocfg) 
            should be placed in a subfolder with this name inside the "data" directory.
        sim_duration : int, optional
            The duration of the simulation in number of steps (default: 500).
        vehicle_prob : float, optional
            The probability of vehicle appearance at each step (default: 0.05).
        pedestrian_prob : float, optional
            The probability of pedestrian appearance at each step (default: 0.05).
        prob_variation : float, optional
            The relative variation range for the probability of appearance for each lane. A value of 1 means that vehicle_prob and pedestrian_prob
            can vary between 0 and 2 times their initial value (default: 0.15).
        seed : int, optional
            A seed for the random number generator for reproducibility (default: None).
        use_seed_every_episode : bool, optional
            Whether to use the seed for every episode (each call to the reset method will reset the seed so that every episode is reproducible),
            or to use the seed only once (default: True). You can use the method reset_seed to reset the seed manually.
        """

        self.MIN_SWITCH = 5 # Minimum duration of a phase (green / red) before switching (in steps). Must be superior to the yellow phase duration.
        self.MIN_GAP = 2 # Minimum gap between vehicles (in meters)
        self.YEL_DURATION = 3 # Duration of the yellow phase (in steps)

        self.sim_duration = sim_duration
        self.vehicle_prob = vehicle_prob
        self.pedestrian_prob = pedestrian_prob
        self.prob_variation = prob_variation

        self.seed = seed
        self.use_seed_every_episode = use_seed_every_episode

        data_dir = "data" if os.path.exists("data") else "Sim/data"
        self.sumocfg = f"{data_dir}/{sim_name}/{sim_name}.sumocfg"

        self.routes, self.ped_routes = extract_routes(f"{data_dir}/{sim_name}/{sim_name}.rou.xml")

    def reset(self, render=False):
        """
        Reset the environment.

        Parameters
        ----------
        render : bool, optional
            Whether to render the simulation for this episode (default: False). 
            If True, the SUMO GUI will be launched, and you will need to press 
            the "Play" button to start the simulation and close the window manually.

        Returns
        -------
        observation : np.array
            The observation of the environment.
        """

        if traci.isLoaded():
            traci.close()

        if hasattr(self, 'np_random'):
            if self.use_seed_every_episode:
                self.np_random = np.random.RandomState(self.seed)
        else:
            if self.seed is not None:
                self.np_random = np.random.RandomState(self.seed)
            else:
                self.np_random = np.random.RandomState()

        sumoBinary = "sumo-gui" if render else "sumo"
        sumoCmd = [sumoBinary, "-c", self.sumocfg]
        traci.start(sumoCmd)

        self.current_step = 0
        self.phase_duration = np.zeros(traci.trafficlight.getIDCount()) # Duration of the current phase (green / red) for each traffic light
        self.yel_step = [-1] * traci.trafficlight.getIDCount() # Current step of the yellow phase, used to count the duration of the yellow phase, -1 if the yellow phase is not active

        self.vehicle_probs_adjusted = [
            max(0, self.vehicle_prob * (1 + self.np_random.uniform(-self.prob_variation, self.prob_variation)))
            for _ in self.routes
        ]

        self.pedestrian_probs_adjusted = [
            max(0, self.pedestrian_prob * (1 + self.np_random.uniform(-self.prob_variation, self.prob_variation)))
            for _ in self.ped_routes
        ]

        self.generate_vehicles()
        self.generate_pedestrians()

        accumulated_waiting_times = self.get_accumulated_waiting_times()
        self.last_waiting_time = sum(accumulated_waiting_times) / 100.0

        return self.get_observation()
    
    def reset_seed(self, seed=None):
        """
        Reset the seed of the environment.

        Parameters
        ----------
        seed : int, optional
            The seed to use for the random number generator (default: None).
        """

        if seed is not None:
            self.seed = seed
            self.np_random = np.random.RandomState(self.seed)
        else:
            if self.seed is not None:
                self.np_random = np.random.RandomState(self.seed)
            else:
                self.np_random = np.random.RandomState()

    def close(self):
        """
        Close the environment.
        """
        traci.close()
    
    def get_observation(self):
        """
        Returns an observation of the environment.

        The observation is a concatenated vector containing the following information for each traffic light:

        - phase_one_hot :
            A one-hot encoded vector indicating the current active green phase.
        - min_green :
            A binary variable indicating whether `MIN_SWITCH` steps have already passed in the current phase.
        - lane_densities :
            A vector containing the number of vehicles in the lane divided by its total capacity for each incoming lane.
        - lane_queues :
            A vector containing the number of queued (speed below 0.1 m/s) vehicles in the lane divided by its total capacity for each incoming lane.

        Returns
        -------
        np.ndarray
            The concatenated observation vector.
        """
 
        "TODO: Extract calculations that can be extracted (especially traci calls) in the reset method to improve the performance of the simulation"
        
        traffic_light_ids = traci.trafficlight.getIDList()
        observation = []

        for i, tl_id in enumerate(traffic_light_ids):
            phase_state = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            current_phase_index = traci.trafficlight.getPhase(tl_id)
            num_phases = len(phase_state)

            phase_one_hot = np.zeros(num_phases)
            phase_one_hot[current_phase_index] = 1
            min_green = self.phase_duration[i] >= self.MIN_SWITCH

            controlled_links = traci.trafficlight.getControlledLinks(tl_id)

            lane_densities = []
            lane_queues = []
            for link in controlled_links:
                incoming_lane = link[0][0]
                
                lane_length = traci.lane.getLength(incoming_lane)
                num_vehicles = traci.lane.getLastStepVehicleNumber(incoming_lane)
                
                mean_vehicle_length = traci.lane.getLastStepLength(incoming_lane)
                lane_capacity = lane_length / (mean_vehicle_length + self.MIN_GAP)
    
                lane_density = num_vehicles / lane_capacity
                lane_densities.append(lane_density)
                
                # Compute queue length (vehicles with speed < 0.1 m/s)
                queue_length = traci.lane.getLastStepHaltingNumber(incoming_lane) / lane_capacity
                lane_queues.append(queue_length)
            
            observation_single_tl = np.concatenate((
                phase_one_hot, 
                [min_green], 
                lane_densities, 
                lane_queues
            ))
            
            observation.append(observation_single_tl)
        
        return np.concatenate(observation)

    def get_action_space(self):
        """
        Returns the action space of the environment.

        We only consider red-green phases for the traffic lights. The action space is a vector 
        with as many elements as there are traffic lights. Each element represents the action 
        for a traffic light, where 0 corresponds to no change and 1 corresponds to switching 
        to the opposite color.

        Returns
        -------
        np.ndarray
            The action space of the environment.
        """

        traffic_light_ids = traci.trafficlight.getIDList()
        action_space = np.zeros(len(traffic_light_ids))

        return action_space 

    def get_observation_space(self):
        """
        Returns a detailed view of the observation space, including the global start and end indices 
        for each element (phase_one_hot, min_green, lane_densities, and lane_queues) for each traffic light.

        See the `get_observation` method for more details on the observation space.

        This method helps to easily extract specific information from the observation vector 
        for each traffic light in the environment.

        Returns
        -------
        dict
            A dictionary containing the total shape and per-traffic-light index ranges for each observation element.
        """
        
        traffic_light_ids = traci.trafficlight.getIDList()
        
        observation_space_infos = {
            "total_shape": None,    # Total shape of the full observation vector for all traffic lights
            "infos_per_TL": []      # List of dictionaries containing index ranges for each traffic light
        }
        
        total_shape = 0
        for tl_id in traffic_light_ids:
            num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
            num_links = len(traci.trafficlight.getControlledLinks(tl_id))

            shape_per_TL = num_phases + 1 + 2 * num_links  # phase_one_hot, min_green, lane_densities, lane_queues
            
            phase_one_hot_start = total_shape
            phase_one_hot_end = phase_one_hot_start + num_phases
            
            min_green_start = phase_one_hot_end
            min_green_end = min_green_start + 1
            
            lane_densities_start = min_green_end
            lane_densities_end = lane_densities_start + num_links
            
            lane_queues_start = lane_densities_end
            lane_queues_end = lane_queues_start + num_links
            
            total_shape += shape_per_TL

            observation_space_infos["infos_per_TL"].append({
                "phase_one_hot": (phase_one_hot_start, phase_one_hot_end),
                "min_green": (min_green_start, min_green_end),
                "lane_densities": (lane_densities_start, lane_densities_end),
                "lane_queues": (lane_queues_start, lane_queues_end)
            })
        
        observation_space_infos["total_shape"] = (total_shape,)
        
        return observation_space_infos


    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : np.ndarray
            The action to take in the environment. See the `get_action_space` method for more details.

        Returns
        -------
        observation : np.ndarray
            The observation of the environment.
        reward : float
            The reward of the environment.
        done : bool
            Whether the episode is finished.
        info : dict
            Additional information about the environment.
        """

        traffic_light_ids = traci.trafficlight.getIDList()

        for i, tl_id in enumerate(traffic_light_ids):
            if action[i] == 1:
                current_phase = traci.trafficlight.getPhase(tl_id)
                num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(tl_id, next_phase)
                self.phase_duration[i] = 0
                self.yel_step[i] = 0
            else:
                if self.yel_step[i] < self.YEL_DURATION:
                    if self.yel_step[i] != -1:
                        self.yel_step[i] += 1
                    self.phase_duration[i] += 1
                else:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tl_id, next_phase)
                    self.phase_duration[i] = 0
                    self.yel_step[i] = -1

        traci.simulationStep()
        self.current_step += 1
        self.generate_vehicles()
        self.generate_pedestrians()

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.current_step >= self.sim_duration

        return observation, reward, done, {}
    
    def generate_vehicles(self):
        """
        Generate vehicles in the simulation. A vehicle is generated with a certain probability on each entry edge.
        The route is chosen randomly among available routes that start with the entry edge.
        """
        for entry_index, entry_routes in enumerate(self.routes):
            if self.np_random.random() < self.vehicle_probs_adjusted[entry_index]:
                if entry_routes:
                    chosen_route = self.np_random.choice(entry_routes)
                    vehicle_id = f'vehicle_{chosen_route}_{self.current_step}'
                    traci.vehicle.add(vehicle_id, chosen_route)
    
    def generate_pedestrians(self):
        """
        Generate pedestrians in the simulation. A pedestrian is generated with a certain probability on each entry edge.
        The route is chosen randomly among available pedestrian routes that start with the entry edge.
        """
        for entry_index, entry_ped_routes in enumerate(self.ped_routes):
            if self.np_random.random() < self.pedestrian_probs_adjusted[entry_index]:
                if entry_ped_routes:
                    chosen_index = self.np_random.randint(0, len(entry_ped_routes))
                    chosen_ped_route = entry_ped_routes[chosen_index]
                    pedestrian_id = f'pedestrian_{chosen_ped_route}_{self.current_step}'
                    last_edge = chosen_ped_route[-1]
                    last_edge_ped = traci.lane.getLength(last_edge + "_0")
                    traci.person.add(pedestrian_id, chosen_ped_route[0], pos=0, depart=self.current_step)
                    traci.person.appendWalkingStage(personID=pedestrian_id, edges=chosen_ped_route, arrivalPos=last_edge_ped)

    def get_reward(self):
        """
        Compute the reward of the environment.

        The reward is computed as the difference between the previous accumulated waiting time 
        and the current accumulated waiting time of vehicles in the simulation.

        Returns
        -------
        float
            The reward of the environment.
        """

        accumulated_waiting_times = self.get_accumulated_waiting_times()
        current_waiting_time = sum(accumulated_waiting_times) / 100.0
        reward = self.last_waiting_time - current_waiting_time
        self.last_waiting_time = current_waiting_time
        return reward
    
    def get_accumulated_waiting_times(self):
        """
        Returns a list of accumulated waiting times for all lanes coming to each traffic lights controlled by the controller.
        """

        traffic_light_ids = traci.trafficlight.getIDList()
        
        accumulated_waiting_times = []
        
        for tl_id in traffic_light_ids:
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
            
            for link in controlled_links:
                lane_id = link[0][0]
                lane_waiting_time = traci.lane.getWaitingTime(lane_id)
                accumulated_waiting_times.append(lane_waiting_time)
        
        return accumulated_waiting_times
    

def extract_routes(file_path):
    """
    Extracts the routes and pedestrian routes from a .rou.xml file, organizing them 
    by their starting edges.

    Note
    ----
    All edges in the simulation must have an associated pedestrian lane for this to work. 
    Pedestrians will always follow the same lanes as vehicles here.

    Parameters
    ----------
    file_path : str
        The path to the .rou.xml file.

    Returns
    -------
    tuple
        A tuple containing:
        - routes : list
            A list of lists, where each inner list contains route names starting with the same edge.
        - ped_routes : list
            A 3D list where the first dimension groups routes by starting edge, the second dimension 
            groups edges by individual routes, and the third dimension lists edges for each route.
    """
    
    tree = ET.parse(file_path)
    root = tree.getroot()

    grouped_routes = defaultdict(list)
    grouped_ped_routes = defaultdict(list)

    for route in root.findall('route'):
        route_id = route.get('id')
        edges = route.get('edges').split()

        if edges:
            start_edge = edges[0]
            grouped_routes[start_edge].append(route_id)
            grouped_ped_routes[start_edge].append(edges)

    routes = [group for group in grouped_routes.values()]
    ped_routes = [group for group in grouped_ped_routes.values()]

    return routes, ped_routes
