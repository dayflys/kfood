import os
from ._dataclass import *

class kfood:
    def __init__(self,root,health=False):
        self.train_set = []
        self.test_set = []
        self.class_weight = []
        if not health:
            self.NUM_TRAIN_FOOD = 42
            self.TRAIN_PATH = os.path.join(root,'kfood_train/train')
            self.TEST_PATH = os.path.join(root,'kfood_val/val')
        else:
            self.NUM_TRAIN_FOOD = 13
            self.TRAIN_PATH = os.path.join(root,'kfood_health_train')
            self.TEST_PATH = os.path.join(root,'kfood_health_val')

        labels = {}
        num_food = [0 for _ in range(self.NUM_TRAIN_FOOD)]
        num_sample = 0

        for subdir, _, files in os.walk(self.TRAIN_PATH):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    f = os.path.join(subdir, file)
                    food = f.split(os.path.sep)[-2]


                    try: labels[food]
                    except:
                        labels[food] = len(labels.keys())

                    # init item
                    item = Food_TrainItem(path=f, food=food, label=labels[food])
                    self.train_set.append(item)
                    num_sample += 1
                    num_food[labels[food]] += 1

        for n in num_food:
            self.class_weight.append(num_sample / n)

        labels = {}
        # test_set
        for root, _, files in os.walk(os.path.join(self.TEST_PATH)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    f = os.path.join(root, file)
                    food = f.split(os.path.sep)[-2]
                    
                    try: labels[food]
                    except:
                        labels[food] = len(labels.keys())
                        
                    item = Food_TestItem(path=f, key=labels[food])
                    self.test_set.append(item)
