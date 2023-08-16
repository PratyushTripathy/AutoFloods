# ifmiap/__init__.py

import os

file_path = os.path.realpath(__file__)
print('Echoing from here:', file_path)

def my_func(n):
    return n ** 3