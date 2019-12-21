import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, data_loader, model, optimizer, head_list, criterions, lambdas, args):
        self.train_loader = data_loader['train']
        self.model = model
        self.optimizer = optimizer
        
        self.head_list = head_list
        self.criterions = criterions
        self.lambdas = lambdas
        
        self.epochs = args.epochs
        
        if args.continue_from is None:
            self.model_dir = os.path.dirname(args.continue_from)
            package = torch.load(args.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.start_epoch = package['epoch']
        else:
            self.continue_from = None
            self.model_dir = args.model_dir
            self.start_epoch = 0
    
    def train(self):
        head_list = self.head_list
        
        for epoch in range(self.start_epoch, self.epochs):
            for iteration, (train_images, target) in enumerate(self.train_loader):
                outputs = self.model(train_images)
                
                estimated_output = {head: None for (head, num_in_features, head_module) in head_list}
                
                for output in outputs:
                    for head in output.keys():
                        if estimated_output[head] is None:
                            estimated_output[head] = output[head].unsqueeze(dim=0)
                        else:
                            estimated_output[head] = torch.cat((estimated_output[head], output[head].unsqueeze(dim=0)), dim=0)
                
                loss = 0

                for (head, num_out_features, head_module) in head_list:
                    loss = loss + self.lambdas[head] * self.criterions[head](estimated_output[head], target[head], target['pointmap'], target['num_object'])

                print("[Epoch {}] iteration {}/{}, loss: {}".format(epoch+1, iteration+1, len(self.train_loader), loss.item()))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, "model_{}.pth".format(epoch+1))
            package = {'state_dict': self.model.state_dict(), 'epoch': epoch+1}
            torch.save(package, model_path)
    
    
class Evaluater(object):
    def __init__(self, data_loader, model, head_list, args):
        self.train_loader = data_loader['test']
        self.model = model
        
        self.head_list = head_list
        
        self.out_image_dir = args.out_image_dir
        
        os.makedirs(self.out_image_dir, exist_ok=True)
        
    def eval(self):
        batch_size = self.train_loader.batch_size
        head_list = self.head_list
        
        self.model.eval()
        
        with torch.no_grad():
            for iteration, (test_images, image_id) in enumerate(self.train_loader):
                outputs = self.model(test_images)
                
                estimated_output = {head: None for (head, num_in_features, head_module) in head_list}
                
                for output in outputs:
                    for head in output.keys():
                        if estimated_output[head] is None:
                            estimated_output[head] = output[head].unsqueeze(dim=0)
                        else:
                            estimated_output[head] = torch.cat((estimated_output[head], output[head].unsqueeze(dim=0)), dim=0)

                for (head, num_out_features, head_module) in head_list:
                    for batch_id in range(batch_size):
                        if head == 'heatmap':
                            plt.figure()
                            plt.imshow(estimated_output[head][-1][batch_id, 0].clone().cpu())
                            # (['heatmap'][stack_id][batch_id, num_channels])
                            save_path = os.path.join(self.out_image_dir, image_id[batch_id] + '_heatmap.png')
                            plt.savefig(save_path)
                            
                            plt.figure()
                            test_image = test_images[batch_id].clone().cpu().permute(1, 2, 0)
                            test_image = ((test_image+1)/2*255).int()
                            plt.imshow(test_image)
                            save_path = os.path.join(self.out_image_dir, image_id[batch_id] + '.png')
                            plt.savefig(save_path)
                        
                return
