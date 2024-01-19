from argument import argument
from data.kfood import kfood
from data.dataset import TrainDataset,TestDataset
from data.dataloader import DataLoaders
from model.SE_Resnet34 import SEResent34
from model.resnet34 import ResNet34
from model.resnet import ResNet18
from model.SE_Resnet import SEResnet
from framework.DANN import DANN
from train.trainHandler import ExpHandler
import torch 
from loss.CCE import CCE, DANNLoss
import random
import numpy as np
import os 
import log
import torchvision.transforms as transforms

def set_experiment(args):
    #set_experiment
    # os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda_visible_devices']
    random.seed(args['rand_seed'])
    np.random.seed(args['rand_seed'])
    torch.manual_seed(args['rand_seed'])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    

def exp(transform=None):
    args, system_args, experiment_args = argument()
    set_experiment(args)
    if not os.path.exists(os.path.join('/workspace/results',args['project'],args['tags'])):
        os.makedirs(os.path.join('/workspace/results',args['project'],args['tags']),exist_ok=True)
    
    builder = log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
    builder.use_local_logger(args['path_log'])
    builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
    logger = builder.build()
    logger.log_arguments(experiment_args)

    domain1_datafile  = kfood(args['path'],True)
    domain2_datafile = kfood(args['path'])

    #preprocessing
    transform = transform
    train_set = domain1_datafile.train_set + domain2_datafile.train_set
    test_set= domain1_datafile.test_set + domain2_datafile.test_set
    
    # dataset
    if transform is list :    
        train_dataset = TrainDataset(train_set,transform[0])
        test_dataset = TestDataset(test_set,transform[1])
    else:
        train_dataset = TrainDataset(train_set,transform)
        test_dataset = TestDataset(test_set,transform)
        
        

    # dataloader
    loader = DataLoaders(args,train_dataset,test_dataset)
    loader = DataLoaders(args,train_dataset,test_dataset)
    train_loader = loader.get_train_dataloader()
    test_loader = loader.get_test_dataloader()
    


    # model
    if args['model_type'] == 0:
        model = ResNet18(args['embedding_size'])
    elif args['model_type'] == 1:
        model = SEResent34(args['embedding_size'])
    elif args['model_type'] == 2:
        model = ResNet34(args['embedding_size'])
    elif args['model_type'] == 3:
        model = SEResnet(args['embedding_size'])
    

    #loss
    args['num_classes'] = domain1_datafile.NUM_TRAIN_FOOD + domain2_datafile.NUM_TRAIN_FOOD
    
    cls_class_weight = np.append(domain1_datafile.class_weight,domain2_datafile.class_weight)
    cls_class_weight = list(cls_class_weight/args['num_classes'])
    
    total_num = domain1_datafile.num_sample + domain2_datafile.num_sample
    disc_class_weight = [domain1_datafile.num_sample/total_num,domain2_datafile.num_sample/total_num]
    
    loss = DANNLoss(args['embedding_size'],args['num_classes'],2,cls_class_weight,disc_class_weight)

    framework = DANN(
        model=model,
        criterion=loss
    )

    framework.cuda()

    # optimizer
    optimizer = torch.optim.AdamW(
        framework.get_parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay'],
        amsgrad=True
    )
    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args['T_0'],
        T_mult=args['T_mult'],
        eta_min=args['lr_max']
    )

    experiment = ExpHandler(args,logger,framework,train_loader,test_loader,optimizer,lr_scheduler)

    experiment.run()

    best_state = experiment.get_state()
    experiment.infer(best_state)
    
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    exp(transform)
