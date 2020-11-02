# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
#added
from utils import progress_bar
import logging
import models

class BANUpdater(object):
    def __init__(self,**kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.alpha=kwargs['alpha']
        self.num_class=kwargs['num_class']
        self.last_model = None
        self.gen = 0

    def update(self, epoch,trainloader,criterion,kld_criterion):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        train_kld_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            if self.gen == 0:
                loss = torch.mean(criterion(outputs, targets))
                kld_loss=0
            else:
                with torch.no_grad():
                    teacher_outputs = self.last_model(inputs).detach()
                kld_loss=kld_criterion(outputs,teacher_outputs)
                ce_loss=torch.mean(criterion(outputs,targets))
                loss=self.alpha*ce_loss+(1-self.alpha)*kld_loss
            train_loss+=loss.item()
            train_kld_loss+=kld_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | kld: %.3f '
                         % (train_loss / (batch_idx + 1), 100. * 0, 0, 0,
                            train_kld_loss / (batch_idx + 1)))
        logger = logging.getLogger('train')
        logger.info('[GEN {}] [Epoch {}] [Loss {:.3f}] [cls {:.3f}] '.format(
            self.gen,
            epoch,
            train_loss / len(trainloader),
            train_kld_loss / len(trainloader),
            ))
        return train_loss / len(trainloader),train_kld_loss / len(trainloader)

    def register_last_model(self,weight,device_ids):
        self.last_model=models.load_model(self.model_name, self.num_class).cuda()
        self.last_model=torch.nn.DataParallel(self.last_model,device_ids=device_ids)
        self.last_model.load_state_dict(weight)
        return

    def kd_loss(self,):
        return

    def __model(self):
        return

    def __last_model(self):
        return

    def __gen(self):
        return


class EBANUpdater(object):
    def __init__(self,**kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.alpha=kwargs['alpha']
        self.num_class=kwargs['num_class']
        self.last_model = None
        self.gen = 0

    def update(self, epoch,trainloader,criterion,kld_criterion):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        train_kld_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            if self.gen == 0:
                loss = torch.mean(criterion(outputs, targets))
                kld_loss=0
            else:
                ensemble_teacher_outputs = torch.zeros(outputs.size()).cuda()
                model_lst = self.last_model
                for i in range(len(model_lst)):
                    with torch.no_grad():
                        teacher_outputs = model_lst[i](inputs).detach()
                    ensemble_teacher_outputs += teacher_outputs
                ensemble_teacher_outputs=ensemble_teacher_outputs/len(model_lst)
                kld_loss=kld_criterion(outputs,ensemble_teacher_outputs)
                ce_loss=torch.mean(criterion(outputs,targets))
                loss=self.alpha*ce_loss+(1-self.alpha)*kld_loss
            train_loss+=loss.item()
            train_kld_loss+=kld_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | kld: %.3f '
                         % (train_loss / (batch_idx + 1), 100. * 0, 0, 0,
                            train_kld_loss / (batch_idx + 1)))
        logger = logging.getLogger('train')
        logger.info('[GEN {}] [Epoch {}] [Loss {:.3f}] [cls {:.3f}] '.format(
            self.gen,
            epoch,
            train_loss / len(trainloader),
            train_kld_loss / len(trainloader),
            ))
        return train_loss / len(trainloader),train_kld_loss / len(trainloader)

    def register_last_model(self,weight,device_ids):
        model_lst = []
        for i in range(len(weight)):
            model = models.load_model(self.model_name, self.num_class).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.load_state_dict(weight[i])
            model_lst.append(model)
        self.last_model = model_lst

    def kd_loss(self,):
        return

    def __model(self):
        return

    def __last_model(self):
        return

    def __gen(self):
        return


class WEBANUpdater(object):
    def __init__(self,**kwargs):
        self.model = kwargs.pop("model")
        self.hypernetwork = kwargs.pop("hypernetwork")
        self.optimizer = kwargs.pop("optimizer")
        self.hypernetwork_optimizer = kwargs.pop("hypernetwork_optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.alpha=kwargs['alpha']
        self.num_class=kwargs['num_class']
        self.last_model = None
        self.gen = 0

    def update(self, epoch,trainloader,criterion,kld_criterion):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        self.hypernetwork.train()
        train_loss = 0
        train_kld_loss = 0
        train_hypernetwok_ce_loss=0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            # hypernetwork
            weights = self.hypernetwork(outputs)  # [64,2]
            #print('weights size is :{}'.format(weights.size()))
            if self.gen == 0:
                loss = torch.mean(criterion(outputs, targets))
                kld_loss=0
                hypernetwork_ce_loss=0
            else:
                ensemble_teacher_outputs = torch.zeros(outputs.size()).cuda()
                model_lst = self.last_model
                for i in range(len(model_lst)):
                    #model_lst[i].eval()
                    temp_weight = weights[:, i].unsqueeze(1).expand(outputs.size())
                    with torch.no_grad():
                        teacher_outputs = model_lst[i](inputs).detach()
                        #print('BAN-{} loss is : {}'.format(i,torch.mean(criterion(teacher_outputs,targets))))
                    ensemble_teacher_outputs += teacher_outputs*temp_weight
                #divide a constant
                ensemble_teacher_outputs = ensemble_teacher_outputs / len(model_lst)
                kld_loss=kld_criterion(outputs,ensemble_teacher_outputs)
                ce_loss=torch.mean(criterion(outputs,targets))
                #print('ce loss is : {}'.format(ce_loss))
                hypernetwork_ce_loss=torch.mean(criterion(ensemble_teacher_outputs,targets))
                #print('ensemble ce loss is : {}'.format(hypernetwork_ce_loss))
                loss=self.alpha*ce_loss+(1-self.alpha)*kld_loss+self.alpha*hypernetwork_ce_loss
            train_loss+=loss.item()
            train_kld_loss+=kld_loss
            train_hypernetwok_ce_loss+=hypernetwork_ce_loss
            self.optimizer.zero_grad()
            self.hypernetwork_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hypernetwork_optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | hyper_ce: %.3f | kld: %.3f '
                         % (train_loss / (batch_idx + 1), hypernetwork_ce_loss / (batch_idx + 1),
                            train_kld_loss / (batch_idx + 1)))
        logger = logging.getLogger('train')
        logger.info('[GEN {}] [Epoch {}] [Loss {:.3f}] [kld {:.3f}] [hypernetwork_ce_loss {:.3f}]'.format(
            self.gen,
            epoch,
            train_loss / len(trainloader),
            train_kld_loss / len(trainloader),
            train_hypernetwok_ce_loss / len(trainloader),
            ))
        return train_loss / len(trainloader),train_kld_loss / len(trainloader),train_hypernetwok_ce_loss / len(trainloader)

    def register_last_model(self,weight,device_ids):
        model_lst = []
        for i in range(len(weight)):
            model = models.load_model(self.model_name, self.num_class).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.load_state_dict(weight[i])
            model_lst.append(model)
        self.last_model = model_lst

    def kd_loss(self,):
        return

    def __model(self):
        return

    def __last_model(self):
        return

    def __gen(self):
        return


class AWEBANUpdater(object):
    def __init__(self,**kwargs):
        self.model = kwargs.pop("model")
        self.hypernetwork = kwargs.pop("hypernetwork")
        self.optimizer = kwargs.pop("optimizer")
        self.hypernetwork_optimizer = kwargs.pop("hypernetwork_optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.alpha=kwargs['alpha']
        self.num_class=kwargs['num_class']
        self.last_model = None
        self.gen = 0

    def update(self, epoch,trainloader,criterion,kld_criterion):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        self.hypernetwork.train()
        train_loss = 0
        train_kld_loss = 0
        train_hypernetwok_ce_loss=0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #inputs, targets = inputs.cuda(), targets.cuda()
            inputs[0], targets,inputs[1] = inputs[0].cuda(), targets.cuda(),inputs[1].cuda()
            outputs_current = self.model(inputs[0])
            # hypernetwork
            weights = self.hypernetwork(outputs_current)  # [64,2]
            #print('weights size is :{}'.format(weights.size()))
            if self.gen == 0:
                loss = torch.mean(criterion(outputs_current, targets))
                kld_loss=0
                hypernetwork_ce_loss=0
            else:
                ensemble_teacher_outputs = torch.zeros(outputs_current.size()).cuda()
                model_lst = self.last_model
                for i in range(len(model_lst)):
                    #model_lst[i].eval()
                    temp_weight = weights[:, i].unsqueeze(1).expand(outputs_current.size())
                    with torch.no_grad():
                        teacher_outputs = model_lst[i](inputs[1]).detach()
                        #print('BAN-{} loss is : {}'.format(i,torch.mean(criterion(teacher_outputs,targets))))
                    ensemble_teacher_outputs += teacher_outputs*temp_weight
                #divide a constant
                ensemble_teacher_outputs = ensemble_teacher_outputs / len(model_lst)
                kld_loss=kld_criterion(outputs_current,ensemble_teacher_outputs)
                ce_loss=torch.mean(criterion(outputs_current,targets))
                #print('ce loss is : {}'.format(ce_loss))
                hypernetwork_ce_loss=torch.mean(criterion(ensemble_teacher_outputs,targets))
                #print('ensemble ce loss is : {}'.format(hypernetwork_ce_loss))
                loss=self.alpha*ce_loss+(1-self.alpha)*kld_loss+self.alpha*hypernetwork_ce_loss
            train_loss+=loss.item()
            train_kld_loss+=kld_loss
            train_hypernetwok_ce_loss+=hypernetwork_ce_loss
            self.optimizer.zero_grad()
            self.hypernetwork_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.hypernetwork_optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | hyper_ce: %.3f | kld: %.3f '
                         % (train_loss / (batch_idx + 1), hypernetwork_ce_loss / (batch_idx + 1),
                            train_kld_loss / (batch_idx + 1)))
        logger = logging.getLogger('train')
        logger.info('[GEN {}] [Epoch {}] [Loss {:.3f}] [kld {:.3f}] [hypernetwork_ce_loss {:.3f}]'.format(
            self.gen,
            epoch,
            train_loss / len(trainloader),
            train_kld_loss / len(trainloader),
            train_hypernetwok_ce_loss / len(trainloader),
            ))
        return train_loss / len(trainloader),train_kld_loss / len(trainloader),train_hypernetwok_ce_loss / len(trainloader)

    def register_last_model(self,weight,device_ids):
        model_lst = []
        for i in range(len(weight)):
            model = models.load_model(self.model_name, self.num_class).cuda()
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model.load_state_dict(weight[i])
            model_lst.append(model)
        self.last_model = model_lst

    def kd_loss(self,):
        return

    def __model(self):
        return

    def __last_model(self):
        return

    def __gen(self):
        return


class BANUpdater_structure(object):
    def __init__(self,**kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.model_name = kwargs['model_name']
        self.alpha=kwargs['alpha']
        self.num_class=kwargs['num_class']
        self.last_model = None
        self.gen = 0

    def update(self, epoch,trainloader,criterion,kld_criterion):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        train_kld_loss = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            if self.gen == 0:
                loss = torch.mean(criterion(outputs, targets))
                kld_loss=0
            else:
                with torch.no_grad():
                    teacher_outputs = self.last_model(inputs).detach()
                kld_loss=kld_criterion(outputs,teacher_outputs)
                ce_loss=torch.mean(criterion(outputs,targets))
                loss=self.alpha*ce_loss+(1-self.alpha)*kld_loss
            train_loss+=loss.item()
            train_kld_loss+=kld_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | kld: %.3f '
                         % (train_loss / (batch_idx + 1), 100. * 0, 0, 0,
                            train_kld_loss / (batch_idx + 1)))
        logger = logging.getLogger('train')
        logger.info('[GEN {}] [Epoch {}] [Loss {:.3f}] [cls {:.3f}] '.format(
            self.gen,
            epoch,
            train_loss / len(trainloader),
            train_kld_loss / len(trainloader),
            ))
        return train_loss / len(trainloader),train_kld_loss / len(trainloader)

    def register_last_model(self,weight,device_ids,model_name):
        self.last_model=models.load_model(model_name, self.num_class).cuda()
        self.last_model=torch.nn.DataParallel(self.last_model,device_ids=device_ids)
        self.last_model.load_state_dict(weight)
        return

    def kd_loss(self,):
        return

    def __model(self):
        return

    def __last_model(self):
        return

    def __gen(self):
        return

