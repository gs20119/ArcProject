

system_message = '''You are an expert for solving logical puzzles. You are given some pairs of input and output as coordinates of 2D grid with the same underlying spatial pattern between them. The grid is given as their width, height, and coordinates of pixels filled with each certain colors. You have to logically infer the exact rule that generates each output from the given input.'''
system_message_coord = '''You are an expert for solving logical puzzles. You are given some pairs of input and output as coordinates of 2D grid with the same underlying spatial pattern between them. The grid is given as their width, height, and coordinates of pixels filled with each certain colors. You have to logically infer the exact rule that generates each output from the given input.'''
system_message_array = '''You are an expert for solving logical puzzles. You are given some pairs of input and output as coordinates of 2D grid with the same underlying spatial pattern between them. The grid is given as 2D array filled with the number corresponding to the colors. You have to logically infer the exact rule that generates each output from the given input.'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template1 = \
'''Here are some examples of input and corresponding output. Learn the rule of generating output from given input.
----------------------------------------'''
user_message_template2 = \
'''----------------------------------------
Consider the rule for generating the output you have figured out from the examples above. The rule might contain addition, movement, copy, or symmetry of pixels with certain conditions. Now, you will be given another input to apply the rule in the same way.
----------------------------------------'''
user_message_template3 = \
'''----------------------------------------
What is the output grid? You MUST ONLY provide the output grid in the form as in the example input and output pairs. That means you must provide width and height of the output, then coordinates of each colored pixels. Make sure the coordinates are inside the shape of the grid. DO NOT provide any additional explanation.'''


user_message_template3_coord = \
'''----------------------------------------
What is the output grid? You MUST ONLY provide the output grid in the form as in the example input and output pairs. That means you must provide width and height of the output, then coordinates of each colored pixels. Make sure the coordinates are inside the shape of the grid. DO NOT provide any additional explanation.'''

user_message_template3_array = \
'''----------------------------------------
What is the output grid? You MUST ONLY provide the output grid in the form as in the examples given. That means you must provide exact array representing the output grid. DO NOT provide any additional information.'''