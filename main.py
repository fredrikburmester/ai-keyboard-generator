# Create starting population of the 1st generation 
import argparse
from functools import cache
import random
import re
import time
import matplotlib.pyplot as plt

simple_example = False

# fingers_interface = {
#   0: 'left_pinky',
#   1: 'left_ring',
#   2: 'left_middle',
#   3: 'left_index',
#   4: 'thumb', 
#   5: 'right_index',
#   6: 'right_middle',
#   7: 'right_ring',
#   8: 'right_pinky'
# }

fingers_interface = {
  'left_pinky': 0,
  'left_ring': 1,
  'left_middle': 2,
  'left_index': 3,
  'thumb': 4,
  'right_index': 5,
  'right_middle': 6,
  'right_ring': 7,
  'right_pinky': 8
}

original_keyboard = "qazwsxedcrfvtgbyhnujmik,ol.pöåä"

def init_population(pop_size):
  population = []
  # Initialize population with random layouts
  for i in range(pop_size):
    keyboard = random.sample(original_keyboard, len(original_keyboard))

    # Convert keyboard list into string
    keyboard_str = ""
    for char in keyboard:
      keyboard_str += char

    population.append(keyboard_str)
  return population

# Create the next generation of layouts
def next_generation(population, pop_size):
  new_gen = []

  # Copy the best 10% of the population to the next generation
  for i in range(int(pop_size*0.1)):
    # check if the layout is already in the new generation
    if population[i] not in new_gen:
      new_gen.append(population[i])

  # Combine the keyboards from the top 50% of the generation
  # and add the new keyboard to the next generation
  for i in range(int(pop_size/2)):
    p1 = random.choice(population[:int(pop_size*0.5)])
    p2 = random.choice(population[:int(pop_size*0.5)])
    child = mate(p1, p2)

    # Convert child list into string
    child_str = ""
    for char in child:
      child_str += char

    # check if the layout is already in the new generation
    if child_str not in new_gen:
      new_gen.append(child)

  # Add random keyboards to the next generation
  for i in range(int(pop_size*0.4)):
    keyboard = random.sample(original_keyboard, len(original_keyboard))

    # Convert keyboard list into string
    keyboard_str = ""
    for char in keyboard:
      keyboard_str += char

    new_gen.append(keyboard_str)

  return new_gen

def mate(keyboard1, keyboard2):
  const_length = len(keyboard1)
  child = ['_' for i in range(const_length)]

  # Add half the keys from keyboard 1
  for i in range(int(const_length/2)):
    child[i] = keyboard1[i]

  for i in range(len(child)):
    if child[i] == '_':
      key = keyboard2[i]
      if key not in child:
        child[i] = key

  # Add missing keys from keyboard 2
  for i in range(len(child)):
    if child[i] == '_':
      for j in range(len(keyboard2)):
        if keyboard2[j] not in child:
          child[i] = keyboard2[j]
          break

  # Sometimes, mutate the child
  prob = random.random()
  if prob > 0.9:
    point1 = random.randint(0, const_length-1)
    point2 = random.randint(0, const_length-1)
    allele1 = child[point1]
    allele2 = child[point2]
    child[point1] = allele2
    child[point2] = allele1

  # Convert child list into string
  child_str = ""
  for char in child:
    child_str += char

  return child_str

@cache
def get_key_row(key):
  if key in ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']: 
    return 1
  elif key in ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']:
    return 2
  elif key in ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.']:
    return 3

@cache
def get_key_col(key):
  if key in ['q', 'a', 'z']:
    return 1
  elif key in ['w', 's', 'x']:
    return 2
  elif key in ['e', 'd', 'c']:
    return 3
  elif key in ['r', 'f', 'v']:
    return 4
  elif key in ['t', 'g', 'b']:
    return 5
  elif key in ['y', 'h', 'n']:
    return 6
  elif key in ['u', 'j', 'm']:
    return 7
  elif key in ['i', 'k', ',']:
    return 8
  elif key in ['o', 'l', '.']:
    return 9
  elif key in ['p', 'ö']:
    return 10
  elif key in ['å', 'ä']:
    return 11

@cache
def get_known_distance(key1, key2):
  adj_distance = 1
  diag_up_left_distance = 1.032
  diag_down_right_distance = 1.118
  diag_up_right_distance = 1.118
  diag_up_left_two_rows_distance = 2.138
  
  # Special distances
  left_index_f_to_t_distance = 1.247
  left_index_g_to_r_distance = 1.605
  left_index_f_to_b_distance = 1.803
  left_index_r_to_b_distance = 2.661
  left_index_t_to_v_distance = 2.015
  left_index_t_to_b_distance = 3.019
  left_index_v_to_g_distance = diag_down_right_distance

  right_index_h_to_u_distance = 1.247
  right_index_j_to_y_distance = 1.605
  right_index_h_to_m_distance = 1.803
  right_index_y_to_m_distance = 2.661
  right_index_u_to_n_distance = 2.015

  right_pinky_ä_to_p_distance = right_index_j_to_y_distance
  right_pinky_ö_to_å_distance = right_index_h_to_u_distance

  if key1 == key2:
    return 0

  if key1 == 'f' and key2 == 't':
    return left_index_f_to_t_distance
  elif key1 == 'g' and key2 == 'r':
    return left_index_g_to_r_distance
  elif key1 == 'f' and key2 == 'b':
    return left_index_f_to_b_distance
  elif key1 == 'r' and key2 == 'b':
    return left_index_r_to_b_distance
  elif key1 == 't' and key2 == 'v':
    return left_index_t_to_v_distance
  elif key1 == 't' and key2 == 'b':
    return left_index_t_to_b_distance
  elif key1 == 'h' and key2 == 'u':
    return right_index_h_to_u_distance
  elif key1 == 'j' and key2 == 'y':
    return right_index_j_to_y_distance
  elif key1 == 'h' and key2 == 'm':
    return right_index_h_to_m_distance
  elif key1 == 'y' and key2 == 'm':
    return right_index_y_to_m_distance
  elif key1 == 'u' and key2 == 'n':
    return right_index_u_to_n_distance
  elif key1 == 'ö' and key2 == 'å':
    return right_index_h_to_u_distance
  elif key1 == 'j' and key2 == 'n':
    return diag_up_right_distance
  elif key1 == 'j' and key2 == 'm':
    return diag_up_right_distance
  elif key1 == 'ä' and key2 == 'p':
    return right_pinky_ä_to_p_distance
  elif key1 == 'ö' and key2 == 'å':
    return right_pinky_ö_to_å_distance
  elif key1 == 'v' and key2 == 'g':
    return left_index_v_to_g_distance
  
  return -1

@cache
def get_distance(key1, key2):
  adj_distance = 1
  diag_up_left_distance = 1.032
  diag_down_right_distance = 1.118
  diag_up_right_distance = 1.118
  diag_up_left_two_rows_distance = 2.138

  dis = get_known_distance(key1, key2)

  if dis != -1:
    return dis
  else:
    dis = get_known_distance(key2, key1)
    if dis != -1:
      return dis

  # If not found, calculate distance

  # Get the column and row of the keys
  key1_col = get_key_col(key1)
  key1_row = get_key_row(key1)
  key2_col = get_key_col(key2)
  key2_row = get_key_row(key2)

  # Get the distance between the keys
  if key1_col == key2_col:
    row_distance = abs(key1_row - key2_row)
    if row_distance == 1:
      return diag_up_left_distance
    elif row_distance == 2:
      return diag_up_left_two_rows_distance
  elif key1_row == key2_row:
    col_distance = abs(key1_col - key2_col)
    return col_distance * adj_distance

  # assertion error if we get here
  assert False, "No distance found for keys: " + key1 + " and " + key2

def euclidean_key_distance(key1, key2):
    # Define the keyboard layout
    keyboard = [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.']
    ]

    # Find the keys on the keyboard
    for row in keyboard:
        if key1 in row:
            key1_row = keyboard.index(row)
            key1_col = row.index(key1)
        if key2 in row:
            key2_row = keyboard.index(row)
            key2_col = row.index(key2)

    # Calculate the distance between the keys
    distance = ((key1_row - key2_row)**2 + (key1_col - key2_col)**2)**0.5

    return distance

def finger_hey_relation(key, fingers = 4):
  if fingers == 4:
    relations = {
      'left_pinky': ['q', 'a', 'z'],
      'left_ring': ['w', 's', 'x'],
      'left_middle': ['e', 'd', 'c'],
      'left_index': ['r', 'f', 'v', 't', 'g', 'b'],
      'right_index': ['y', 'h', 'n', 'u', 'j', 'm'],
      'right_middle': ['i', 'k', ','],
      'right_ring': ['o', 'l', '.'],
      'right_pinky': ['p', 'å', 'ä', 'ö']
    }
  elif fingers == 3:
    relations = {
      'left_index': ['r', 'f', 'v', 't', 'g', 'b'],
      'left_middle': ['e', 'd', 'c'],
      'left_ring': ['w', 's', 'x'],
      'right_index': ['u', 'h', 'j', 'y', 'n', 'm'],
      'right_middle': ['i', 'k', 'o', 'l', 'ö', 'ä', 'å', '.', ','],
    }
  elif fingers == 1:
    relations = {
      'left_index': ['q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r', 'f', 'v', 't', 'g', 'b'],
      'right_index': ['u', 'h', 'j', 'y', 'n', 'm', 'i', 'k', 'o', 'l', 'ö', 'ä', 'å', '.', ','],
    }

  for finger in relations:
    if key in relations[finger]:
      return fingers_interface[finger]

def eval(keyboard, text, distance_calculation = 'Simple'):
  # Finger positions
  fingers = ['a', 's', 'd', 'f','_', 'j', 'k', 'l', 'ö']
  # three_fingers = ['s', 'd', 'f','_', 'j', 'k', 'l']
  # two_fingers = ['d', 'f','_', 'j', 'k']
  # one_finger = ['f','_', 'j']
  
  # For each letter in the text, check what finger should be used, check where the finger is, and calculate the distance
  distance = 0

  for i in range(len(text)):
    if simple_example:
      print()
      print("Letter: " + text[i])

    key_index = keyboard.find(text[i])
    key = original_keyboard[key_index]

    if simple_example: print("Key: " + key + " index: " + str(key_index))

    finger = finger_hey_relation(key)

    if simple_example: print(f"{fingers[finger]} -> {key}, distance: {euclidean_key_distance(fingers[finger], key)}")

    finger_pos = fingers[finger]
    if distance_calculation == 'Simple':
      distance += euclidean_key_distance(finger_pos, key)
    elif distance_calculation == 'Complex':
      distance += get_distance(finger_pos, key)

    fingers[finger] = key

  return distance

def print_keyboard(keyboard):
  # Print every third character on a new line
  for i in range(len(keyboard) - 3):
    if i % 3 == 0:
      print(keyboard[i], end='')
  print(keyboard[29])

  print(" ", end='')
  
  # Shift the keyboard string 1 character to the right
  keyboard = keyboard[1:] + keyboard[0]

  for i in range(len(keyboard) - 3):
    if i % 3 == 0:
      print(keyboard[i], end='')
  print(keyboard[29])

  print("  ", end='')

  # Shift the keyboard string 1 character to the right
  keyboard = keyboard[1:] + keyboard[0]

  for i in range(len(keyboard) - 4):
    if i % 3 == 0:
      print(keyboard[i], end='')
  
  print()
  

def main():
  # command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--text", help="text to be typed")
  parser.add_argument("-e", "--example", help="run a simple example", action="store_true")
  parser.add_argument("-g", "--generations", help="number of generations", type=int)
  parser.add_argument("-p", "--population", help="population size", type=int)
  args = parser.parse_args()

  global simple_example
  simple_example = args.example

  if args.population:
    pop_size = args.population
  else: 
    pop_size = 100

  if args.generations:
    generations = args.generations
  else:
    generations = 100

  # Import the exaluation text from the text file text.txt
  if args.text:
    evaluation_text = args.text
  else:
    evaluation_text = open("dataset.txt", "r", encoding="utf8").read()

  population = init_population(pop_size)

  # Convert to lower case
  evaluation_text = evaluation_text.lower()

  # Remove all non-alphabetical characters
  evaluation_text = re.sub('[^a-zåäö.,]', '', evaluation_text)

  # Replace all spaces with underscore
  evaluation_text = evaluation_text.replace(" ", "_")

  all_best_evals = []

  # Plot the best evaluation for each generation
  plt.xlabel('Generation')
  plt.ylabel('Fitness')
  plt.title('Best fitness score per generation')

  # Simple Example --------
  if simple_example: 
    generations = 1
    pop_size = 1
    population = [original_keyboard]
    if args.text:
      evaluation_text = args.text
    else:
      evaluation_text = open("test_dataset.txt", "r").read()
    print(eval(original_keyboard, "hejäqa"))
    print(eval(original_keyboard, evaluation_text))
  # -----------------------

  for i in range(generations):
    evals = []

    start = time.time()
    for j in range(len(population)):
      evals.append(eval(population[j], evaluation_text))
    end = time.time()

    print("--------------------")
    print("Generation: " + str(i+1) )
    print("Population length: ", len(population))
    print("Evaluation time: " + str(end - start))
    print("Best eval: ", evals[0])

    all_best_evals.append(int(evals[0]))
    
    # Best 10 keyboards
    if pop_size > 3:
      print("\nBest keyboards from previous generation:")
      for k in range(3):
        print(f" - {population[k]}")
      print()
      print("Best keyboard:\n")
      print_keyboard(population[0])
    else: 
      print("Best keyboard from previous generation:\n")
      print_keyboard(population[0])

    print()
    population = [x for _,x in sorted(zip(evals, population))]
    next_gen = next_generation(population, pop_size)
    population = next_gen

    # Save the plot to a file
    plt.plot(all_best_evals, color="green")
    plt.savefig(f"./images/plot.png")

  if not simple_example:
    print("\nBest keyboard: ")
    print_keyboard(population[0])
    plt.show()

if __name__ == "__main__":
  main()

