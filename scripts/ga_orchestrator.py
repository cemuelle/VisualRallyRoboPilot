import os
import json
from ga_pipeline import genAl

# parameters
GENERATIONS = 20 # number of generations per iteration
PITY = 20 # number of instructions to add at the end of the individual, it serves as a "with a few more instructions, it would have passed the gate" preventer


def optimize_trajectories(base_directory):
    for context_dir in sorted(os.listdir(base_directory)):
        # for each context directory in the base directory
        context_path = os.path.join(base_directory, context_dir)
        if os.path.isdir(context_path):
            print(f"Processing context: {context_dir}")
            
            for iteration_dir in sorted(os.listdir(context_path)):
                # for each iteration directory in the current context directory
                iteration_path = os.path.join(context_path, iteration_dir)
                if os.path.isdir(iteration_path):
                    print(f"Processing iteration: {iteration_dir}")
                    
                    # Collect all individuals (JSON files) in the current iteration directory and aggregate all the individuals in a population list
                    population = []
                    context = {}
                    for file in sorted(os.listdir(iteration_path)):
                        if file.endswith('.json') and file.startswith('individual'):
                            file_path = os.path.join(iteration_path, file)
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    # checks if the context is the same for all individuals
                                    if len(context) == 0:
                                        # if first, sets context
                                        context = data["initial_context"]
                                        # formats the controls for the ga
                                        tupledControls = []
                                        for i in data["controls"]:
                                            tupledControls.append(tuple(i))
                                        # adds the pity controls by duplicating the last controls {PITY} times
                                        for i in range(PITY):
                                            tupledControls.append(tupledControls[-1])
                                        # adds the individual to the population
                                        population.append((iteration_path, tupledControls))
                                    elif context == data["initial_context"]:
                                        # if not the first individual but the context matches
                                        # formats the controls for the ga
                                        tupledControls = []
                                        for i in data["controls"]:
                                            tupledControls.append(tuple(i))
                                        # adds the pity controls by duplicating the last controls {PITY} times 
                                        for i in range(PITY):
                                            tupledControls.append(tupledControls[-1])
                                        # adds the individual to the population
                                        population.append((iteration_path, tupledControls))
                                    else:
                                        # if the context doesn't match
                                        raise Exception(f"The context doesn't match the previous context {context} ({file_path})")

                            except json.JSONDecodeError as e:
                                print(f"    Error decoding JSON in file {file_path}: {e}")
                            except Exception as e:
                                print(f"    Error processing file {file_path}: {e}")

                    # prints the resulting context and population
                    print("Finished with interation ", iteration_path)
                    print(context)
                    print(population)
                    # pass it through the GA pipeline
                    gaOut, scores, bests = genAl(GENERATIONS,population,context)

                    # prints all the elits outputed by the GA in the right directory
                    for elit in range(len(gaOut)):
                        individual_file_path = os.path.join(iteration_path, f"ga_augmented_{iteration_dir}{elit}.json")
                        with open(individual_file_path, "w") as f:
                            json.dump({
                                "initial_context": context,
                                "controls": gaOut[elit]
                            }, f, indent=4)   

                    # stores the mean scores of the population at each generation
                    individual_result_path = os.path.join(iteration_path, f"improvement_{iteration_dir}.txt")
                    with open(individual_result_path, "w") as f:
                        f.write(str(scores))

                    # stores the best score of the population at each generation
                    generational_best_path = os.path.join(iteration_path, f"bests_{iteration_dir}.txt")
                    with open(generational_best_path, "w") as f:
                        f.write(str(bests))

        

base_directory = "out_red"  # Directory with the controls
optimize_trajectories(base_directory)
