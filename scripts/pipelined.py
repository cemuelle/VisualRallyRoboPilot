


import os
import pickle
import lzma
import glob
import numpy as np
import json
import random
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from data_collector import DataCollectionUI
from data_collector_evaluate_pilot import DataCollectionEvaluatePilot
from PyQt6 import QtWidgets
import sys
import torch
import torch.nn as nn
import json

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = self.dropout(self.sigmoid(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = MLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load("models/MLP_model.pth", weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def nn_infer(self, message):
        X = torch.tensor([list(message.raycast_distances) + [message.car_speed]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X)

        output_list = output.tolist()[0]
        formatted_output = ["{:.4f}".format(x) for x in output_list] 

        forward = float(formatted_output[0]) > 0.5
        backward = float(formatted_output[1]) > 0.5
        left = float(formatted_output[2]) > 0.5
        right = float(formatted_output[3]) > 0.5

        car_speed = message.car_speed

        if not forward and not backward and abs(car_speed) < 0.3:
            if car_speed < 0:
                backward = True
            else:
                forward = True

        return [
            ("forward", forward),
            ("back", backward),
            ("left", left),
            ("right", right)
        ]

    def process_message(self, message, data_collector):

        car_position = message.car_position

        if data_collector.gate.is_car_through((car_position[0], car_position[2])) and len(data_collector.recorded_data) > 2:
            # data_collector.saveRecord(close_after_save=True)
            data_collector.network_interface.disconnect()
            QtWidgets.QApplication.quit()
        else:
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)

def get_mlp_path(initial_position, initial_angle, initial_speed, gate_position):
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionEvaluatePilot(
        nn_brain.process_message,
        initial_position=initial_position,
        initial_angle=initial_angle,
        initial_speed=initial_speed,
        record=True,
        record_image=False
    )
    data_window.gate.set_gate(gate_position[0], gate_position[1], gate_position[2])
    app.exec()

    return data_window.recorded_data


# Load gate configurations from JSON file
with open("gates_simple_track.json", "r") as file:
    gate_configurations = json.load(file)["gate_position_simple"]

# Predefined variables for gates
gate_0_output = []
gate_1_output = []
gate_2_output = [] 
gate_3_output = []  
gate_4_output = []  
gate_5_output = []  


# Mapping gate index to variable
gate_outputs = {
    0: gate_0_output,
    1: gate_1_output,
    2: gate_2_output,
    3: gate_3_output,
    4: gate_4_output,
    5: gate_5_output,
}

# Process each gate
for idx, gate_config in enumerate(gate_configurations):
    gate_output = gate_outputs[idx]
    
    # Shared parameters
    initial_position = gate_config["start_position"]
    initial_angle = gate_config["start_orientation"]
    gate_position = (
        gate_config["p1_gate"],
        gate_config["p2_gate"],
        5  # Fixed gate width
    )
    
    # Generate multiple individuals for the gate
    for individual_idx in range(10):  # 10 individuals per gate
        initial_speed = 30  # Example variation in initial speed
        
        # Generate recorded data for the individual
        recorded_data = get_mlp_path(initial_position, initial_angle, initial_speed, gate_position)
        
        # Format the recorded data
        formatted_data = (
            f"individual_{idx}_{individual_idx}",
            [tuple(snapshot.current_controls) for snapshot in recorded_data]
        )
        gate_output.append(formatted_data)


def nControls(ind):
    a,w,s,d = 0,0,0,0
    controls = ind[0][1]

    for i in controls:
        na,nw,ns,nd = i
        a += na
        w += nw
        s += ns
        d += nd

    return a,w,s,d

def elitism(pop, elitism_count):
    elites = []
        # Select top elite_count individuals from the sorted population (from individual_with_scores)
    sorted_pop = sorted(pop, key=lambda x: x[2], reverse=False)

    for i in range(min(elitism_count, len(sorted_pop))):
        individual_id, controls, fitness = sorted_pop[i]

        # Create the same structure as the child individuals (name, controls)
        elites.append((individual_id,controls,fitness))

    for i in range(elitism_count):
        print("la fine equipe: ")
        print("   ", i, ": ")
        print(elites[i][2])
        print(nControls(elites[i]))
        print(elites[i][1])
    # print("elites: ", elites, "\n")
    return elites

def add_fitness(individual_controls):
    updated_population = []
    for idx, individual in enumerate(individual_controls):
        if isinstance(individual, tuple):
            if len(individual) == 3:  # Already has fitness
                individual_id, controls, _ = individual
            elif len(individual) == 2:  # Initial population
                individual_id, controls = individual
            else:
                print(f"Unexpected tuple length at index {idx}: {individual}")
                continue
        else:
            print(f"Unexpected entry at index {idx}: {individual}")
            continue

        # Generate a dummy fitness score between 0 and 1
        fitness = random.uniform(0, 1)

        # Update the individual tuple with the fitness value
        updated_population.append((individual_id, controls, fitness))

    return updated_population

def crossover(parent1, parent2):
    """
    Perform crossover between two parents by randomly choosing control elements from each parent.

    Parameters:
    - parent1: List of control tuples for the first parent.
    - parent2: List of control tuples for the second parent.

    Returns:
    - child: New child created by selecting control elements from parent1 and parent2.
    """
    child = []

    for c1, c2 in zip(parent1, parent2):
        # For each control element, randomly choose from parent1 or parent2
        chosen_parent = random.choice([c1, c2])
        child.append(chosen_parent)

    return child

def add_crossover_pop(pop, population_size, elite_count):

    new_pop=[]

    # Create new individuals using crossover until the population size is met
    while len(new_pop) < (population_size - elite_count):
        # Select two random parents from sorted population, ensuring they are not the same
        parent1_data = random.choice(pop)
        parent2_data = random.choice(pop)

        # Ensure parent1 and parent2 are not the same individual
        while parent1_data == parent2_data:
            parent2_data = random.choice(pop)
        # Unpack the controls from the parents
        parent1_name, parent1_controls, parent1_fitness = parent1_data  # Unpack name and controls
        parent2_name, parent2_controls, parent2_fitness = parent2_data
        # Perform crossover between the parents
        child_controls = crossover(parent1_controls, parent2_controls)

        # Create the child with a random fitness score (this could be calculated differently)
        #child_fitness_score = random.random()

        # Store the child as a tuple (individual_name, controls, fitness_score)
        #new_pop.append(('child_' + str(len(new_pop) + 1), child_controls, child_fitness_score))
        new_pop.append(('child_' + str(len(new_pop) + 1), child_controls))

    return new_pop

def mutate(individual, mutation_rate):
    """
    Mutate the individual's controls based on the mutation_rate.
    This will mutate each control with a certain probability, setting each control value to 0 or 1.

    Parameters:
    - individual: A tuple of (name, controls)
    - mutation_rate: Probability of mutation per control in each individual

    Returns:
    - individual: The mutated individual with updated controls
    """

    name, controls = individual  # Get the controls from the individual

    newControls = []

    for i in range(len(controls)):
        # for each control
        mutatedControls = []
        for j in range (4):
            # for each WASD control
            if np.random.rand() < mutation_rate:
                # if the mutation happens (random number)
                mutatedControls.append(0 if controls[i][j] == 1 else 1)
            else:
                mutatedControls.append(controls[i][j])
        newControls.append(tuple(mutatedControls))

    return (name, newControls)  # Return the mutated individual

def smoothingTemplate(individual, mutation_rate, directions):
    """
    Template for the smoothing mutators. apply smoothing patterns through a convolution (size 3) on the controls. (for example, W at times 0 to 2 are [1,1,0], so the smoothing sets it to [1,1,1])
    The patterns can smooth towards 1 or 0, it's decided by the "control" input tab.

    Parameters:
    - individual: A tuple of (name, controls)
    - mutation_rate: Probability of mutation per control in each individual
    - directions: A list of values between (-1, 0 and 1) for each 4 controls (WSAD), corresponding to the smoothing direction:
        -1 => no smoothing
        0  => smoothing toward 0
        1  => smoothing toward 1

    Returns:
    - individual: The mutated individual with updated controls

    current smoothing patterns:
    a, a, b => a, a, a   |   a, b, a => a, a, a   |   a, b, a => a, a, b   |   b, a, a => a, a, a
    """

    name, controlsRaw = individual
    controls = list(map(list, zip(*controlsRaw)))  # transpose the controls to facilitate working with a window

    for j in range(4):
        # for each WASD control
        a = directions[j]
        if a == 0 or a == 1:
            b = 1 - a
            patterns = [([a,a,b],[a,a,a]), ([a,b,a],[a,a,a]), ([a,b,a],[a,a,b]), ([b,a,a],[a,a,a])] # for each pattern, if the window matches the first list and the mutation happen, replaces the window by the pattern second list
            for i in range(len(controls[0])-2):
                # for each control (minus 2 since we move a window of size 3)
                commands = [controls[j][i],controls[j][i+1],controls[j][i+1]] # gets the window
                for k in patterns:
                    # for each pattern
                    if commands == k[0]:
                        if np.random.rand() < mutation_rate:
                            # if the window matches the pattern and the mutation happens (random number)
                            controls[j][i],controls[j][i+1],controls[j][i+2] = k[1]
                            #print("turned ", commands, " into ", k[1], " boss. (", j, ", ", i, ")")

    reshapedControls = list(map(list, zip(*controls)))
    addingTuples = []
    for i in reshapedControls:
        addingTuples.append(tuple(i))
    return [(name, addingTuples)]  # Return the mutated individual

def mutateFaster(individual, mutation_rate):
    """
    Mutate the individual's controls based on the mutation_rate.
    This will compare successive values for an input and try and smooth it, here, only the W.

    Parameters:
    - individual: A tuple of (name, controls)
    - mutation_rate: Probability of mutation per control in each individual

    Returns:
    - individual: The mutated individual with updated controls

    current smoothing patterns:
    a, a, b => a, a, a   |   a, b, a => a, a, a   |   a, b, a => a, a, b   |   b, a, a => a, a, a
    """
    return smoothingTemplate(individual, mutation_rate, [1,-1,-1,-1])

def mutateTurner(individual, mutation_rate):
    """
    Mutate the individual's controls based on the mutation_rate.
    This will compare successive values for an input and try and smooth it, here, only the W (towards 0) and A-D (towards 1).

    Parameters:
    - individual: A tuple of (name, controls)
    - mutation_rate: Probability of mutation per control in each individual

    Returns:
    - individual: The mutated individual with updated controls

    current smoothing patterns:
    a, a, b => a, a, a   |   a, b, a => a, a, a   |   a, b, a => a, a, b   |   b, a, a => a, a, a
    """
    return smoothingTemplate(individual, mutation_rate, [0,-1,1,1])

def mutateRandomSmooth(individual, mutation_rate):
    """
    Mutate the individual's controls based on the mutation_rate.
    This will compare successive values for an input and try and smooth it, here, according to 4 random values.

    Parameters:
    - individual: A tuple of (name, controls)
    - mutation_rate: Probability of mutation per control in each individual

    Returns:
    - individual: The mutated individual with updated controls

    current smoothing patterns:
    a, a, b => a, a, a   |   a, b, a => a, a, a   |   a, b, a => a, a, b   |   b, a, a => a, a, a
    """
    return smoothingTemplate(individual, mutation_rate, [random.randrange(-1, 2),random.randrange(-1, 2),random.randrange(-1, 2),random.randrange(-1, 2)])

def mutate_population(population, mutation_rate):
    """
    Mutates the individuals in the population based on mutation_rate.
    """

    mutated_population = []

    for individual in population:
        method = random.randrange(0, 3)
        match method:
            case 0:
                mutated_individual = mutate(individual, mutation_rate)
            case 1:
                mutated_individual = mutateFaster(individual, mutation_rate)
            case 2:
                mutated_individual = mutateTurner(individual, mutation_rate)
            case 3:
                mutated_individual = mutateRandomSmooth(individual, mutation_rate)
        mutated_population.append(mutated_individual)

    return mutated_population

def preprocess_population(population):
    processed = []
    for individual in population:
        if isinstance(individual, list):  # Handle nested lists
            processed.extend(individual)
        else:
            processed.append(individual)
    return processed


def genetic_algorithm(generation, mutation_rate, population_size, elitism_count, individual_controls):
    # Start the evolution process for the specified number of generations
    for gen in range(generation):
        print(f"\nGeneration {gen + 1}:")
        individual_with_scores = add_fitness(individual_controls)
        print(f"Fitness done {gen + 1}")
        # Step 1: Select Elite individuals
        elite = elitism(individual_with_scores, elitism_count)
        print(f"Elitism done {gen + 1}")
        # Step 2: Create the next generation using crossover
        crossed_pop = add_crossover_pop(individual_with_scores, population_size, elitism_count)
        print(f"crossover done {gen + 1}")
        # Step 3: Mutate the crossed population
        mutated_pop = mutate_population(crossed_pop, mutation_rate)
        print(f"mutated_pop {gen + 1}")
        popi = preprocess_population(mutated_pop)
        # Step 4: Add the elite individuals to the mutated population
        next_generation = elite + popi  # Elite individuals directly pass to the next generation
        individual_controls = next_generation # apply the new generation
        print(f"create final generation {gen + 1}")
        # Step 5: Print the current population after mutation
        '''print("Mutated Population:")
        for individual in next_generation:
            name, controls = individual
            print(f"Individual: {name}, Controls: {controls}")
            print("\n")
        '''
        #print("\n")
        #print(individual_controls)
        print(f"ready for gen {gen + 2}")

    return next_generation

genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_0_output)
genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_1_output)
genetic_algorithm(generation=3, mutation_rate=0, population_size=10, elitism_count=1, individual_controls=gate_2_output)







