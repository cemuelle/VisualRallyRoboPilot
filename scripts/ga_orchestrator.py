import os
import json
from ga_pipeline import genAl

def process_blocks(base_directory):
    for context_dir in sorted(os.listdir(base_directory)):
        context_path = os.path.join(base_directory, context_dir)
        if os.path.isdir(context_path):
            print(f"Processing context: {context_dir}")
            
            for iteration_dir in sorted(os.listdir(context_path)):
                iteration_path = os.path.join(context_path, iteration_dir)
                if os.path.isdir(iteration_path):
                    print(f"Processing iteration: {iteration_dir}")
                    
                    # Collect all individuals (JSON files) in the current iteration block
                    population = []
                    context = {}
                    for file in sorted(os.listdir(iteration_path)):
                        if file.endswith('.json'):
                            file_path = os.path.join(iteration_path, file)
                            try:
                                
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    print(len(context))
                                    if len(context) == 0:
                                        context = data["initial_context"]
                                        tupledControls = []
                                        for i in data["controls"]:
                                            tupledControls.append(tuple(i))

                                        population.append((iteration_path, tupledControls))

                                    elif context == data["initial_context"]:
                                        tupledControls = []
                                        for i in data["controls"]:
                                            tupledControls.append(tuple(i))
                                        population.append((iteration_path, tupledControls))
                                    else:
                                        raise Exception(f"The context doesn't match the previous context {context} ({file_path})") 

                            except json.JSONDecodeError as e:
                                print(f"    Error decoding JSON in file {file_path}: {e}")
                            except Exception as e:
                                print(f"    Error processing file {file_path}: {e}")

                    print("finito le ", iteration_dir)
                    print(context)
                    print(population)
                    gaOut = genAl(2,population,context)
                    for elit in gaOut:
                        individual_file_path = os.path.join(iteration_path, f"ga_augmented_{iteration_dir}.json")
                        with open(individual_file_path, "w") as f:
                            json.dump({
                                "initial_context": context,
                                "controls": elit
                            }, f, indent=4)   

        

base_directory = "out"  # Directory with the controls
process_blocks(base_directory)
