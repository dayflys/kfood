from argument import argument
from data.kfood import kfood
from data.dataset import TrainDataset,TestDataset
from data.dataloader import DataLoaders
from model.SE_Resnet34 import SEResent34
from model.resnet34 import ResNet34
from model.resnet import ResNet18
from model.SE_Resnet import SEResnet
from framework.framework import kfoodframework
from train.trainHandler import ExpHandler
import torch 
from loss.CCE import CCE
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
    

def exp(transform=None,adapt=False):
    args, system_args, experiment_args = argument()
    set_experiment(args)
    if not os.path.exists(os.path.join('/workspace/results',args['project'],args['tags'])):
        os.makedirs(os.path.join('/workspace/results',args['project'],args['tags']),exist_ok=True)
    
    builder = log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
    builder.use_local_logger(args['path_log'])
    builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
    logger = builder.build()
    logger.log_arguments(experiment_args)
    if adapt:
        args['init_model'] = True
    #prepare-data
    if args['init_model']:
        datafile  = kfood(args['path'],True)
    else: datafile = kfood(args['path'])

    #preprocessing
    transform = transform
    args['num_classes'] = datafile.NUM_TRAIN_FOOD

    # dataset
    if transform is list :    
        train_dataset = TrainDataset(datafile.train_set,transform[0])
        test_dataset = TestDataset(datafile.test_set,transform[1])
    else:
        train_dataset = TrainDataset(datafile.train_set,transform)
        test_dataset = TestDataset(datafile.test_set,transform)

    # dataloader
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
    
        
    if args['init_model']:
        model.load_state_dict(torch.load(os.path.join('/workspace/results',args['project'],args['tags'],'model_best.pt'))['model'])

    # loss
    loss = CCE(args['embedding_size'],datafile.NUM_TRAIN_FOOD,datafile.class_weight)

    framework = kfoodframework(
        pre_processing=None,
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
        transforms.RandomHorizontalFlip(),  # 랜덤 수평 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 밝기, 대비, 채도 및 색상 조정
        transforms.RandomRotation(10),  # 랜덤 회전
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 랜덤 크롭 및 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    exp(transform)
    exp(transform,True)