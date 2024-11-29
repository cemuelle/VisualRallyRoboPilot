import os
import pickle
import lzma
import glob
import numpy as np
import json
import random
import numpy as np
from control_car import send_simulation_request
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
# parameters
PROTOCOL = "http"
SERVER_IP = "127.0.0.1"
PORT = 5000

# global variables for parameters set on init
gate_p1 = [-140, -21]
gate_p2 = [-165, -24]
thickness = 5
car_position = [-100, 0, -50]  # x, y, z
car_speed = 50
car_angle = -90


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

    # Evaluate the individual using the simulation
    def send_single_request(list_controls):
        global gate_p1, gate_p2, thickness, car_position, car_speed, car_angle
        #print("hop la les controles: ")
        #print(list_controls)

        if isinstance(list_controls, tuple):
            if len(individual) == 3:  # Already has fitness
                name, controlsRaw, _ = individual
            elif len(individual) == 2:  # Initial population
                name, controlsRaw = individual
            else:
                print(f"Unexpected tuple length at index {idx}: {individual}")
        else:
            print(f"Unexpected entry at index {idx}: {individual}")


        status, time = send_simulation_request(
            protocol=PROTOCOL,
            server_ip=SERVER_IP,
            port=PORT,
            gate_p1=gate_p1,
            gate_p2=gate_p2,
            thickness=thickness,
            car_position=car_position,
            car_speed=car_speed,
            car_angle=car_angle,
            list_controls=controlsRaw,
        )
        fitness = time if status else 100
        return (name,controlsRaw,fitness)

    # Function to execute parallel tasks
    def run_simulation_in_parallel(n, individual_controls):
        results = []

        # Thread pool executor
        with ThreadPoolExecutor() as executor:
            # Submit tasks with thread ID
            futures = [
                executor.submit(send_single_request, individual_controls[thread_id])
                for thread_id in range(n)
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"Error: {e}")

        # Print results and return them
        #print("resultats: ", results)
        return results

    pop = len(individual_controls)
    while(True):
        newPop = run_simulation_in_parallel(pop,individual_controls)
        cnt = 0
        for i in newPop:
            name, controls, score = i
            if score != 100:
                cnt += 1
            if cnt >= 2:
                return newPop
        print("generation with no one passing the finish line, there must have been a problem. let's do it again.")

def nControls(ind):
    a,w,s,d = 0,0,0,0
    controls = individual_controls[0][1]

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

def add_crossover_pop(pop, population_size=20, elite_count=4):

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

def graph_speed_over_generations(generation_data):
    """
    Plots the average speed across generations.

    :param generation_data: List of tuples, where each tuple contains generation number and average speed.
    """
    generations = [data[0] for data in generation_data]
    avg_speeds = [data[1] for data in generation_data]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_speeds, marker='o', linestyle='-', color='b', label='Average Speed')
    plt.title('Average Speed Across Generations')
    plt.xlabel('Generation Number')
    plt.ylabel('Average Speed')
    plt.grid(True)
    plt.legend()
    plt.show()

def genetic_algorithm(generation, mutation_rate, population_size, elitism_count, individual_controls):
    generation_data = []  # Store generation number and average speed
    # Start the evolution process for the specified number of generations
    for gen in range(generation):
        print(f"\nGeneration {gen + 1}:")
        individual_with_scores = add_fitness(individual_controls)
        print(f"Fitness done {gen + 1}")
        # Step 0: Graphs Speed avg
        speeds = [individual[2] for individual in individual_with_scores]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        generation_data.append((gen + 1, avg_speed))
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
    # Plot average speed across generations
    graph_speed_over_generations(generation_data)    

    return next_generation


def genAl(generation, initial_controls, section):
    global gate_p1, gate_p2, thickness, car_angle, car_position, car_speed
    gate_p1, gate_p2,thickness,car_position,car_speed,car_angle = section
    '''
    gate_p1 = [-140,-21]
    gate_p2 = [-165,-24]
    thickness = 5
    car_position = [10,0,1]  # x, y, z
    car_speed = 50
    car_angle = -90
    '''
    return genetic_algorithm(generation=generation, mutation_rate=0.1, population_size=10, elitism_count=1, individual_controls=initial_controls)


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
genetic_algorithm(generation=1, mutation_rate=0.2, population_size=10, elitism_count=1, individual_controls=individual_controls)

"""individual_controls = [('individual_0',
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
    ]"""

#paramTab =   [[-140,-21], [-165,-24], 5, [10,0,1], 50, -90]
#yay = genAl(5,individual_controls,paramTab)