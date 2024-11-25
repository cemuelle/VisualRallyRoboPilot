import os
import pickle
import lzma
import glob
import numpy as np
import json
import random
import numpy as np
from control_car import send_simulation_request



def load_data(path):

    files = glob.glob(path)
    individual_controls = []

    # Iterate through each file (representing one individual)
    for i, file in enumerate(files):
        with lzma.open(file, "rb") as f:
            data = pickle.load(f)
            
            # Collect all current_controls from the records in this file
            controls = [record.current_controls for record in data if hasattr(record, 'current_controls')]
            
            # Store in the list as a tuple (individual_id, controls)
            individual_controls.append((f'individual_{i}', controls))

    # Print the number of individuals and the stored controls
    print(f"Total individuals: {len(individual_controls)}")
    print("\nStored Controls:")

    # Print all the current_controls for each individual
    for individual, controls in individual_controls:
        print(f"{individual}: {controls}")

    return individual_controls

def add_fitness(individual_controls):
    scored_population = []
    for individual_id, controls in individual_controls:
        # Assume some initial parameters for the gate and car
        gate_p1 = [-140,-21]
        gate_p2 = [-165,-24]
        thickness = 5
        car_position = [10,0,1]  # x, y, z
        car_speed = 50
        car_angle = -90

        # Evaluate the individual using the simulation
        status, time = send_simulation_request(
            protocol="http",
            server_ip="127.0.0.1",
            port=5000,
            gate_p1=gate_p1,
            gate_p2=gate_p2,
            thickness=thickness,
            car_position=car_position,
            car_speed=car_speed,
            car_angle=car_angle,
            list_controls=controls,
        )

        # Assign fitness score (inverse of time taken; higher is better)
        fitness = time if status else 0
        scored_population.append((individual_id, controls, fitness))
        print(individual_id,controls,fitness)
        print("\n")

    return scored_population

def elitism(pop, elitism_count):

    elites = []
        # Select top elite_count individuals from the sorted population (from individual_with_scores)
    sorted_pop = sorted(pop, key=lambda x: x[2], reverse=False)

    for i in range(min(elitism_count, len(sorted_pop))):
        individual_id, controls, fitness = sorted_pop[i]
        
        # Create the same structure as the child individuals (name, controls)
        elites.append(individual_id,controls,fitness)
    print(elites)
    return elites

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
    controls = individual[1]  # Get the controls from the individual
    
    # Iterate over each control
    for i in range(len(controls)):
        # If the control is a tuple or list, mutate each element in it
        if isinstance(controls[i], (tuple, list)):
            controls[i] = tuple(
                np.random.choice([0, 1], size=len(controls[i])).tolist()
            )  # Convert to tuple and avoid np.int64
        # If the control is an np.int64 (or similar), mutate it directly
        elif isinstance(controls[i], np.int64):
            controls[i] = int(np.random.choice([0, 1]))  # Convert to standard int
        else:
            # Otherwise, handle as normal, assuming it should be a normal int
            if np.random.rand() < mutation_rate:
                controls[i] = int(np.random.choice([0, 1]))  # Convert to int

    return (individual[0], controls)  # Return the mutated individual

def mutate_population(population, mutation_rate):
    """
    Mutates the individuals in the population based on mutation_rate.
    """
    mutated_population = []
    
    for individual in population:
        mutated_individual = mutate(individual, mutation_rate)
        mutated_population.append(mutated_individual)
    
    return mutated_population

def genetic_algorithm(generation, mutation_rate, population_size, elitism_count):

    
    #individual_con = load_data("data_trajectory_test/*.npz")
    individual_controls = [('individual_0',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_1',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_2',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_3',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_4',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_5',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_6',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_7',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_8',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
        ('individual_9',
        [(1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (1, 0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0)]),
    ]

    
    

    

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
        # Step 4: Add the elite individuals to the mutated population
        next_generation = elite + mutated_pop  # Elite individuals directly pass to the next generation
        print(f"create final generation {gen + 1}")
        # Step 5: Print the current population after mutation
        print("Mutated Population:")
        for individual in next_generation:
            print(f"Individual: {individual[0]}, Controls: {individual[1]}")
            print("\n")
        

        print(f"ready for gen {gen + 2}")

    return next_generation

final_pop = genetic_algorithm(generation=10, mutation_rate=0.8, population_size=10, elitism_count=2)