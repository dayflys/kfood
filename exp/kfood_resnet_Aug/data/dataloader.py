import torch 

class DataLoaders:
    def __init__(self,args,trainset,testset):
        self.args = args
        self.init_train_dataloader(trainset)
        self.init_test_dataloader(testset)

    def init_train_dataloader(self,dataset):
        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.args['num_workers'],
            batch_size=self.args['batch_size'],
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )


    def init_test_dataloader(self,dataset):
        self.test_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.args['num_workers'] ,
            batch_size=1,
            pin_memory=True,
            shuffle=False
            )


    def get_train_dataloader(self):
        return self.train_loader

    def get_test_dataloader(self):
        return self.test_loader