import os 
import itertools

def argument():
    system_args = {
    'path'  : '/workspace/kfood/DB',
    'project': 'kfood',
    'tags': '44',
    'name' : 'kfood_resnet_DANN',
    'wandb_group':'kfood_compare',
    'path_log':'/results',
    'description':'',
    'path_scripts': f'{os.path.dirname(os.path.realpath(__file__))}',
    'wandb_api_key': '6f7ed2ba0891af08e1ccd10a1e28ee8bbf9a7e8d',
    'wandb_entity':'kkwr0504',
    'flag' : True,
    
    }
    experiment_args = {
    
    #system argument
    'epoch' : 100,
    'rand_seed': 1,
    'num_workers': 4,
    'embedding_size': 512,
    'init_model': '',
    'cuda_visible_devices': '1',
    'model_type' : 2,
    # data processing


    #training argument
    'batch_size': 256,
    'lr' : 1e-3,
    'weight_decay': 2e-5,
    'T_0': 101,
    'T_mult': 2,
    'lr_max': 1e-8,
    }
    
    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_log'] = os.path.join(args['path_log'], args['project'], args['name'])
    
    return args, system_args, experiment_args