# offline_searcher.py — Defines CombinedGA for genetic-algorithm-based scenario search.
import random
import math
import numpy as np
from utility import average_population_distance

def _distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

def _sanitize_position_info(pi: dict) -> dict:
    # Backward compatibility: migrate the legacy "surrounding_transforms" field to "surrounding_info".
    if 'surrounding_transforms' in pi and 'surrounding_info' not in pi:
        pi['surrounding_info'] = [{'transform': t, 'type': 'car'} for t in pi['surrounding_transforms']]
        pi.pop('surrounding_transforms', None)
    if 'surrounding_info' in pi:
        pi['vehicle_num'] = len(pi['surrounding_info'])
    return pi

def sample_position_info(carla_map):
    NPC_TYPES = ["bicycle", "car"]
    v_num = random.choice([20])

    spawn_points = carla_map.get_spawn_points()
    while True:
        ego_idx = random.randint(0, len(spawn_points) - 1)
        ego_spawning = spawn_points[ego_idx]
        nearby = [sp for i, sp in enumerate(spawn_points) if i != ego_idx]
        if len(nearby) >= v_num:
            break
    selected_spawns = random.sample(nearby, v_num)
    surrounding_info = [{"transform": sp, "type": random.choice(NPC_TYPES)} for sp in selected_spawns]
    return {
        "vehicle_num": v_num,
        "ego_transform": ego_spawning,
        "surrounding_info": surrounding_info
    }

def sample_action_info(vehicle_num, seq_len=50):
    action_space = np.array([[0,3],[1,3],[2,3],[0,0],[1,0],[2,0]], dtype=np.int32)
    actions_per_vehicle = []
    for _ in range(vehicle_num):
        seq = [action_space[random.randint(0, len(action_space)-1)] for _ in range(seq_len)]
        actions_per_vehicle.append(seq)
    return actions_per_vehicle

class CombinedGA:
    """
    Genetic algorithm for searching adversarial scenario configurations.

    Each individual encodes a position_info dict (ego + surrounding NPC transforms).
    Fitness drives selection; mutation introduces diversity via maximum-distance resampling
    relative to the accumulated safe_set.
    """

    def __init__(self, carla_map, population_size=10, generations=5, seq_len=5, enable_action_ga=False):
        self.carla_map = carla_map
        self.safe_set = []
        self.population_size = population_size
        self.generations = generations
        self.seq_len = seq_len
        self.enable_action_ga = enable_action_ga

        self.population = []  # Each element: {"position_info": {...}, optionally "action_info": [...]}
        self.best_individual = None
        self.best_fitness = float('-inf')

    def sample_initial_population(self):
        for _ in range(self.population_size):
            pos_info = sample_position_info(self.carla_map)
            ind = {"position_info": _sanitize_position_info(pos_info)}
            if self.enable_action_ga:
                ind["action_info"] = sample_action_info(ind["position_info"]["vehicle_num"], self.seq_len)
            self.population.append(ind)

    def crossover_individuals(self, parent1, parent2, crossover_rate=0.8):
        if random.random() >= crossover_rate:
            # No crossover: return sanitized copies of both parents unchanged.
            return (
                {“position_info”: _sanitize_position_info(parent1[“position_info”])},
                {“position_info”: _sanitize_position_info(parent2[“position_info”])}
            )

        p1 = _sanitize_position_info(parent1[“position_info”])
        p2 = _sanitize_position_info(parent2[“position_info”])

        child1 = {'position_info': {}}
        child2 = {'position_info': {}}

        # Swap surrounding NPC sets while keeping each parent's ego transform.
        child1['position_info']['surrounding_info'] = p2['surrounding_info']
        child1['position_info']['vehicle_num'] = len(p2['surrounding_info'])
        child1['position_info']['ego_transform'] = p1['ego_transform']

        child2['position_info']['surrounding_info'] = p1['surrounding_info']
        child2['position_info']['vehicle_num'] = len(p1['surrounding_info'])
        child2['position_info']['ego_transform'] = p2['ego_transform']

        return child1, child2

    def mutation(self, individual, mutation_rate=0.2):
        """
        Mutate an individual by optionally replacing it with the most diverse candidate.

        Samples 10 random candidates and selects the one with the greatest average
        distance from safe_set, maximizing diversity in the search population.
        """
        if random.random() < mutation_rate:
            # Sample 10 candidates and select the one farthest from safe_set.
            candidates = []
            scores = []
            for _ in range(10):
                pos_info = sample_position_info(self.carla_map)
                ind = {"position_info": _sanitize_position_info(pos_info)}
                candidates.append(ind)
            for ind in candidates:
                dist = average_population_distance(ind, self.safe_set)
                scores.append(dist)
            return candidates[int(np.argmax(scores))]
        else:
            # No mutation: return a sanitized copy of the original individual.
            out = {"position_info": _sanitize_position_info(individual["position_info"])}
            return out

    def resample(self):
        pos_info = sample_position_info(self.carla_map)
        return {"position_info": _sanitize_position_info(pos_info)}
