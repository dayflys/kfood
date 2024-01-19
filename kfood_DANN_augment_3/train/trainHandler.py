import torch
from tqdm import tqdm
import numpy as np

class ExpHandler:
    def __init__(self, args,logger, model,train_loader,valid_loader, optimizer, scheduler):
        self.args = args
        self.logger = logger
        self.model = model
        self.valid_loader = valid_loader
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        
        self.best_acc = 0
        self.best_state = model.copy_state_dict()
        

    def run(self):
        self.local_step = 0
        for epoch in tqdm(range(self.args['epoch']), total=self.args['epoch']):
            self.train(self.train_loader,epoch)
            corrects, _, total_samples = self.eval(self.valid_loader)
            accuracy = self.accuracy(corrects, total_samples,epoch)
            self.logger.log_metric('accuracy',accuracy)
            print(f'{epoch} epoch accuracy : {accuracy}')
        self.save()

    def save(self):
        torch.save(self.best_state,f'/workspace/results/{self.args["project"]}/{self.args["tags"]}/model_best.pt')

    def get_state(self):
        return self.best_state

    def infer(self,best_state):
        self.model.load_state_dict(best_state)
        corrects, predicts, total_samples = self.eval(self.valid_loader)
        accuracy = self.accuracy(corrects, total_samples,'infer')
        f1 = self.f1_score(corrects, predicts, total_samples)
        self.logger.log_metric('best_accuracy',accuracy)
        self.logger.log_metric('best_f1',f1)
        

    def train(self, loader,epoch):
        self.model.train()

        count = 0
        cls_loss_sum = 0
        disc_loss_sum = 0
        total = len(loader)
        total_steps = total * self.args['epoch']
        start_steps = total * epoch
        
        with torch.set_grad_enabled(True) and tqdm(total=total, ncols=90) as pbar:
            for idx,(x, label,domain) in enumerate(loader):
                # clear grad
                self.optimizer.zero_grad()
                # feed forward
                p = float(idx + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                x = x.to(dtype=torch.float32, device='cuda')
                label = label.to(dtype=torch.int64, device='cuda')
                domain = domain.to(dtype=torch.int64, device='cuda')
                
                cls_loss,disc_loss = self.model(x,alpha, label,domain)
                # backpropagation
                loss = cls_loss+disc_loss
                loss.backward()
                self.optimizer.step()
                    # log

                count += 1
                cls_loss_sum += cls_loss.item()
                disc_loss_sum += disc_loss.item()
                if len(loader) * 0.02 <= count:
                    self.logger.log_metric("cls_loss", cls_loss_sum / count)
                    self.logger.log_metric("disc_loss", disc_loss_sum / count)
                    count = 0
                    cls_loss_sum = 0
                    disc_loss_sum = 0

                #pbar
                desc = f'{epoch} epoch [(cls_loss):{cls_loss.item():.3f}|(disc_loss):{disc_loss.item():.3f}'
                pbar.set_description(desc)
                pbar.update(1)


    def eval(self,loader):
        # set test mode
        self.model.eval()
        softmax = torch.nn.Softmax(dim=-1)
        
        corrects = [0 for _ in range(self.args['num_classes'])]
        total_samples = [0 for _ in range(self.args['num_classes'])]
        predicts = [0 for _ in range(self.args['num_classes'])]

        with torch.set_grad_enabled(False) and tqdm(total=len(loader), ncols=90) as pbar:
            for x, label,_ in loader:
                # to cuda
                
                x = x.to(dtype=torch.float32, device='cuda')

                # inference
                x = self.model(x)
                x = torch.mean(x, 0,True)
                p = softmax(x)
                p = torch.max(p, dim=1)[1]
                
                # count
                for i in range(p.size(0)):
                    _p = p[i].item()
                    l = label[i].item()
                    predicts[_p] += 1
                    total_samples[l] += 1
                    if _p == l:
                        corrects[l] += 1
                pbar.update(1)
        
        return corrects, predicts, total_samples

    def f1_score(self, corrects, predicts, total_samples):

        f1_sum = 0
        for i in range(len(corrects)):
            if predicts[i] == 0:
                f1 = 0
            else:
                precision = corrects[i]/predicts[i]
                recall = corrects[i]/total_samples[i]
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = (2 * precision * recall) / (precision + recall)
            f1_sum += f1
        return f1_sum / len(corrects)

    def accuracy(self, corrects, total_samples,epoch=None):
        accuracy = sum(corrects) / sum(total_samples) * 100

        if self.best_acc < accuracy:
            self.best_acc = accuracy
            self.best_state = self.model.copy_state_dict()
            if epoch is not None:
                print(f'{epoch}-epoch is the best!')

        return accuracy


    # def plot(self):
    #     plt.title('epoch per acc Graph')
    #     plt.ylabel('accuracy')
    #     x = list(range(0,len(self.acc_list)))
    #     plt.plot(x,self.acc_list)
    #     plt.show()


