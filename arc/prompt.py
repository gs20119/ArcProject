

system_message = '''You are an expert for solving logical puzzles. You are given some pairs of input and output as coordinates of 2D grid with the same underlying spatial pattern between them. The grid is given as their width, height, and coordinates of pixels filled with each certain colors. You have to logically infer the exact rule that transforms each input to the corresponding output.'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template1 = \
'''Here are three examples of input and corresponding output. You have to examine given examples and learn the common specific patterns that transforms each input to the corresponding output.
----------------------------------------'''
user_message_template2 = \
'''----------------------------------------
From examples above, you can think of the following contexts:  
- Compare among the inputs to find any common shapes like lines and rectangles, repeating patterns, copy, rotation, or symmetry. Do the same work among the outputs. 
- If you find any speical structures and colors you found in the examples, think what they indicates.
- Figure out the difference between the input and the corresponding output. 
- If count of pixels for each colors does not change in the output, examine where the shapes in the input moves their position in the output and why. 
- Otherwise, examine where the pixels of certain color added or changed, and think how they are related to the shapes, colors of the input.
----------------------------------------
Remember the rules you figured out. From now, you will be given another input to apply the same rule to generate the output you have figured out from the examples above. 
----------------------------------------'''
user_message_template3 = \
'''----------------------------------------
Answer the description of output grid corresponding to the input in the same format. That means you must provide exact width and height of the output grid, then coordinates of each colored pixels. 
Double check that the rule of transformation you applied exactly aligns with the examples. Also, ensure the coordinates are inside the shape of the grid.
----------------------------------------'''
