import os

import numpy as np
from ._dataclass import *

class kfood:
    def __init__(self,root,health=False):
        self.train_set = []
        self.test_set = []
        self.class_weight = np.array([])
        if not health:
            self.NUM_TRAIN_FOOD = 42
            self.TRAIN_PATH = os.path.join(root,'kfood_train/train')
            self.TEST_PATH = os.path.join(root,'kfood_val/val')
            plus = 0
        else:
            self.NUM_TRAIN_FOOD = 13
            self.TRAIN_PATH = os.path.join(root,'kfood_health_train')
            self.TEST_PATH = os.path.join(root,'kfood_health_val')
            plus = 42

        labels = {}
        self.num_food = [0 for _ in range(self.NUM_TRAIN_FOOD)]
        self.num_sample = 0

        for subdir, _, files in os.walk(self.TRAIN_PATH):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    f = os.path.join(subdir, file)
                    food = f.split(os.path.sep)[-2]


                    try: labels[food]
                    except:
                        labels[food] = len(labels.keys()) + plus
                    
                    domain = 0 if not health else 1
                    
                    # init item
                    item = Food_TrainItem(path=f, food=food, label=labels[food], domain=domain)
                    self.train_set.append(item)
                    self.num_sample += 1
                    self.num_food[labels[food]-plus] += 1

        for n in self.num_food:
            self.class_weight = np.append(self.class_weight,n)
        
        labels = {}
        # test_set
        for root, _, files in os.walk(os.path.join(self.TEST_PATH)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    f = os.path.join(root, file)
                    food = f.split(os.path.sep)[-2]
                    
                    try: labels[food]
                    except:
                        labels[food] = len(labels.keys()) + plus
                    
                    domain = 0 if not health else 1
                    
                    item = Food_TestItem(path=f, key=labels[food], domain=domain)
                    self.test_set.append(item)
