# Create starting population of the 1st generation 
from functools import partial
import random
import re
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

fingers_interface = {
  0: 'left_pinky',
  1: 'left_ring',
  2: 'left_middle',
  3: 'left_index',
  4: 'thumb', 
  5: 'right_index',
  6: 'right_middle',
  7: 'right_ring',
  8: 'right_pinky'
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

  # Add missing keys from keyboard 2
  for i in range(int(const_length/2), const_length):
    for j in range(const_length):
      if keyboard2[j] not in child:
        child[i] = keyboard2[j]
        break

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

def get_key_row(key):
  if key in ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'å']: 
    return 1
  elif key in ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä']:
    return 2
  elif key in ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.']:
    return 3


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
      return adj_distance
    elif row_distance == 2:
      return diag_up_left_two_rows_distance
  elif key1_row == key2_row:
    col_distance = abs(key1_col - key2_col)
    return col_distance * adj_distance

  # assertion error if we get here
  assert False, "No distance found for keys: " + key1 + " and " + key2

def get_key_finger_relation(key):
  row = get_key_row(key)
  col = get_key_col(key)

  if key == '_':
    return 4
  elif key == '.':
    return 7
  elif key == ',':
    return 6

  if col == 1:
    return 0
  elif col == 2:
    return 1
  elif col == 3:
    return 2
  elif col == 4:
    return 3
  elif col == 5:
    return 3
  elif col == 6:
    return 5
  elif col == 7:
    return 5
  elif col == 8:
    return 6
  elif col == 9:
    return 7
  elif col == 10:
    return 8
  elif col == 11:
    return 8
  
  # assertion error if we get here
  assert False, "No key finger relation found"

def eval(keyboard, text):
  # Finger positions
  fingers = ['a', 's', 'd', 'f','_', 'j', 'k', 'l', 'ö']
  
  # For each letter in the text, check what finger should be used, check where the finger is, and calculate the distance
  distance = 0
  for i in range(len(text)):
    # print()
    # print("Letter: " + text[i])

    # Get index of key in new keyboard string
    key_index = original_keyboard.find(text[i])

    # Find the corresponding key in original keyboard
    key = keyboard[key_index]

    # print("Key: " + key + " index: " + str(key_index))

    finger = get_key_finger_relation(key)
    # print(f"{fingers[finger]} -> {key}, distance: {get_distance(fingers[finger], key)}")
    finger_pos = fingers[finger]
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
  print("Original keyboard:")
  print_keyboard(original_keyboard)

  evolutions = 300

  pop_size = 1000
  population = init_population(pop_size)

  # Import the exaluation text from the text file text.txt
  evaluation_text = open("text.txt", "r").read()

  # Convert to lower case
  evaluation_text = evaluation_text.lower()

  # Remove all non-alphabetical characters
  evaluation_text = re.sub('[^a-zåäö.,]', '', evaluation_text)

  # Replace all spaces with underscore
  evaluation_text = evaluation_text.replace(" ", "_")

  all_best_evals = []

  # Plot the best evaluation for each generation

  plt.xlabel('Generation')
  plt.ylabel('Evaluation')
  plt.title('Best evaluation for each generation')


  for i in range(evolutions):
    print("--------------------")
    print("Evolution: " + str(i+1) )
    print("Population length: ", len(population))

    evals = []
    pool = Pool()

    # Time the evaluation
    start = time.time()
    evals = pool.map(partial(eval, text=evaluation_text), population)
    # for j in range(pop_size):
      # evals.append(eval(population[j], evaluation_text))
    end = time.time()

    print("Evaluation time: " + str(end - start))

    print("Best eval: ", evals[0])

    all_best_evals.append(int(evals[0]))

    print("Best keyboard from previous generation:\n")

    print_keyboard(population[0])

    # Best 10 keyboards
    print("\nBest 10 keyboards from previous generation:")
    for k in range(10):
      print(population[k])

    # Sort population by evaluation
    population = [x for _,x in sorted(zip(evals,population))]
    next_gen = next_generation(population, pop_size)
    population = next_gen

    # Save the plot to a file
    plt.plot(all_best_evals)
    plt.savefig(f"./images/plot.png")


  print("\nBest keyboard: ")
  print_keyboard(population[0])
  
  plt.show()

if __name__ == "__main__":
  main()
