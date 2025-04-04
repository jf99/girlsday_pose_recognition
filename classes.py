import os
from config import *

def get_classes():
    filenames = os.listdir(training_dir)
    classes = []
    for filename in filenames:
        if os.path.isdir(os.path.join(os.path.abspath("."), training_dir, filename)):
            classes.append(filename)
    classes.sort()
    return classes