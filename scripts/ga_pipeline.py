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
import matplotlib.pyplot as plt
from kubernetes import client, config

config.load_kube_config()

# parameters
PORT = 5000

# global variables for parameters set on init
gate_p1 = [-20, -10]
gate_p2 = [-20, 10]
thickness = 5
car_position = [60, 0, 0]  # x, y, z
car_speed = 30
car_angle = -90

# function to get all our usable pods ips
def get_pods():
    v1 = client.CoreV1Api()
    pod_list = v1.list_namespaced_pod("isc3",label_selector='tier=headempty-zeb')
    podIps = []
    for pod in pod_list.items:
        podIps.append(pod.status.pod_ip)
    return podIps

# function to add fitness to a population, need the population and a list of ips on which to make simulation requests
def add_fitness(individual_controls, list_ips, n_elits):
    # Evaluate one individual using a pod for the simulation
    def send_single_request(list_controls, ip):
        global gate_p1, gate_p2, thickness, car_position, car_speed, car_angle

        # the inputs can already have a score or not
        if isinstance(list_controls, tuple):
            if len(list_controls) == 3:  # Already has fitness
                name, controlsRaw, _ = list_controls
            elif len(list_controls) == 2:  # Initial population
                name, controlsRaw = list_controls
            else:
                print(f"Unexpected tuple length at index {name}: {controlsRaw}")
        else:
            print(f"Unexpected entry at index {name}: {controlsRaw}")

        # request the score for the current individual
        reussied, status, time, col = send_simulation_request(
            protocol="http",
            server_ip=ip,
            port=5000,
            gate_p1=gate_p1,
            gate_p2=gate_p2,
            thickness=thickness,
            car_position=car_position,
            car_speed=car_speed,
            car_angle=car_angle,
            list_controls=controlsRaw,
        )
        # prints the result
        print(reussied, status, time, col)

        # the fitness is the number of instructions it took to reach the gate + 50 * the number of colisions with walls.
        # If it didn't reach the gate, the fitness is 1000
        fitness = time + 50 * col if status else 1000
        return (name,controlsRaw,fitness)

    # Function to execute parallel tasks
    def run_simulation_in_parallel(n, individual_controls):
        results = []

        # Thread pool executor
        with ThreadPoolExecutor() as executor:
            # Submit tasks with thread ID
            futures = [
                executor.submit(send_single_request, individual_controls[thread_id], list_ips[thread_id])
                for thread_id in range(n)
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)

        # Print results and return them
        print("resultats: ")
        for i in results:
            name, controls, score = i
            print(name, nControls(i), score)
        return results

    
    pop = len(individual_controls)
    while(True):
        # loop to check if enough population went through the gate to make an elite
        newPop = run_simulation_in_parallel(pop,individual_controls)
        cnt = 0 # count of the number of individuals that passed the gates
        tries = 5 # number of attempts before accepting a bad population

        
        for i in newPop:
            # for each individual in scored population
            name, controls, score = i
            if score != 1000:
                # counts the individuals that passed the gate
                cnt += 1

            if cnt >= n_elits or tries <= 0:
                # if we have enough elits or if we have already tried {tries} times, continue
                return newPop
            
        tries -= 1
        print("generation with no one passing the finish line, there must have been a problem. let's do it again.")


# function to represent an individual by the sum of its controls
# mainly a debug function to check if modifications occur each generation, simpler with this than with huge (0-1,0-1,0-1,0-1) lists
def nControls(ind):
    a,w,s,d = 0,0,0,0
    controls = ind[1]

    for i in controls:
        na,nw,ns,nd = i
        a += na
        w += nw
        s += ns
        d += nd

    return a,w,s,d


""" THIS TAKES ALL INDIVIDUAL, SORT THEM BY FITNESS SCORE, AND TAKE THE LOWEST DELTA AS ELITE"""
def elitism(pop, elitism_count):
    elites = []
    sorted_pop = sorted(pop, key=lambda x: x[2], reverse=False)

    for i in range(min(elitism_count, len(sorted_pop))):
        
        individual_id, controls, fitness = sorted_pop[i]

        elites.append((individual_id,controls,fitness))

    """ CONSOLE CONTROL """
    """for i in range(elitism_count):
        print("la fine equipe: ")
        print("   ", i, ": ")
        print(elites[i][2])
        print(nControls(elites[i]))
        print(elites[i][1])"""
    
    """ RETURN X NUMBER OF ELITES """
    return elites

""" HERE WE REPOPULATE BY SELECTING ONLY THE ELITE, 
    THIS ALLOWS US TO REMOVE COMPLETELY FAILED INDIVIDUAL 
     WHICH FAILED THE SIMULATION PROCESS 
     SINCE WE CANNOT CONTROL THE TIMING OF THE SIMULATION
"""
def populate(elites, population_size, elitism_count):
    """
    Creates a population by replicating the elite individual.

    Parameters:
        elite (tuple): A tuple of the form (name, controls, fitness), where:
            - name (str): Identifier for the individual.
            - controls (list): List of control values for the individual.
            - fitness (float): Fitness score of the individual.
        population_size (int): Number of individuals to generate.

    Returns:
        list: A list of replicated individuals, each structured as (name, controls, fitness).
    """
    populated = []

    """ THIS WILL ADD ONLY ELITE INTO THE NEXT GENERATION """
    while len(populated) < population_size - elitism_count:
        for elite in elites:
            if len(populated) < population_size:
                # Append a copy of the elite individual
                populated.append(elite)

    """ CONSOLE CONTROL """
    #print("Population:", populated)
    return populated



""" THIS MUTATION WILL ALLOW EXPLORATION SINCE IT 
    WILL EXPLORE ALL POSSIBILITIES OF CONTROLS EQUALLY"""
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

    name, controls, _ = individual  # Get the controls from the individual

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
    name, controlsRaw, _ = individual
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

""" BASED ON 4 DIFFERENT MUTATION FUNCTIONS 
    WE RANDOMLY CHOOSE A MUTATION AND APPLY 
    TO THE INDIVIDUAL CONTROLS
    
    OUR ELITES ARE THEREFORE SLIGHTLY DIFFERENT 
    FROM EACH OTHER.
    
    OUR ELITES HAVING ALREADY EXPLORED THE TRAJECTORY
    OUR MUTATION WILL HELP FOR THE EXPLOTATION """

def mutate_population(population, mutation_rate):
    """
    Mutates the individuals in the population based on mutation_rate.
    """

    mutated_population = []

    for individual in population:
        method = random.randrange(0, 3)
        match method:
            case 0:
                print("full random mutation")
                mutated_individual = mutate(individual, mutation_rate)
            case 1:
                print("faster mutation")
                mutated_individual = mutateFaster(individual, mutation_rate)
            case 2:
                print("turner mutation")
                mutated_individual = mutateTurner(individual, mutation_rate)
            case 3:
                print("smooth random mutation")
                mutated_individual = mutateRandomSmooth(individual, mutation_rate)
        mutated_population.append(mutated_individual)

    return mutated_population

""" THIS ENSURE DATA FORMAT STAYS CONSISTENT """
# COULD BE OVERKILL
def preprocess_population(population):
    processed = []
    for individual in population:
        if isinstance(individual, list):  # Handle nested lists
            processed.extend(individual)
        else:
            processed.append(individual)
    return processed

""" THIS GRAPHS THE SPEED OF EACH GENERATIONS """
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
    generation_best = []
    game_ips = get_pods()
    # Start the evolution process for the specified number of generations
    for gen in range(generation):
        print(f"\nGeneration {gen + 1}:")
        individual_with_scores = add_fitness(individual_controls,game_ips, elitism_count)
        print(f"Fitness done {gen + 1}")
        # Step 0: store the best of the generation
        speeds_raw = [individual[2] for individual in individual_with_scores]
        speeds = []
        for i in speeds_raw:
            if i != 1000:
                speeds.append(i)
        print(speeds)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        generation_data.append((gen + 1, avg_speed))
        # Step 1: Select Elite individuals
        elite = elitism(individual_with_scores, elitism_count)
        generation_best.append(elite[0][2])
        print(f"Elitism done {gen + 1}")
        # Step 2: Create the next generation using crossover)
        elite_pop = populate(elite, population_size, elitism_count)
        # Step 3: Mutate the crossed population
        mutated_pop = mutate_population(elite_pop, mutation_rate)
        print("size of mutated pop", len(mutated_pop))
        print(f"mutated_pop {gen + 1}")
        popi = preprocess_population(mutated_pop)
        # Step 4: Add the elite individuals to the mutated population
        next_generation = elite + popi  # Elite individuals directly pass to the next generation
        individual_controls = next_generation # apply the new generation
        print(f"create final generation {gen + 1}")
        # Step 5: Print the current population after mutation
        print("final gen size ", len(next_generation))
        print(f"ready for gen {gen + 2}")


    strappedElites = []
    for i in elite:
        strappedElites.append(i[1])
    return strappedElites, generation_data, generation_best


# function to call the whole ga pipeline from outside the file
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
    return genetic_algorithm(generation=generation, mutation_rate=0.1, population_size=10, elitism_count=2, individual_controls=initial_controls)
