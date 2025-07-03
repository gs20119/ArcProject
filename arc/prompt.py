
system_message = '''You are an expert for solving logical puzzles. You are given some pairs of input and output as 2D grid with the same underlying spatial pattern between them. You have to logically infer the exact rule that transforms each input to the corresponding output.'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template1 = \
'''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------'''
user_message_template2 = \
'''----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
----------------------------------------'''

user_message_template3 = \
'''----------------------------------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:'''