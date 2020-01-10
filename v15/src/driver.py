import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import show_heatmap, show_image, get_keypoints, get_inv_camera_matrix

class Trainer(object):
    def __init__(self, data_loader, model, optimizer, head_list, criterions, lambdas, args):
        self.train_loader = data_loader['train']
        self.valid_loader = data_loader['valid']
        
        self.model = model
        self.optimizer = optimizer
        
        self.head_list = head_list
        self.criterions = criterions
        self.lambdas = lambdas
        
        self.epochs = args.epochs
        
        self.train_loss = {head: torch.zeros(100) for head, num_in_features, head_module in head_list} # Can save 100 epochs
        self.valid_loss = {head: torch.zeros(100) for head, num_in_features, head_module in head_list}
        
        if args.continue_from is not None:
            self.model_dir = os.path.dirname(args.continue_from)
            package = torch.load(args.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.start_epoch = package['epoch']
            self.optimizer.load_state_dict(package['optim_dict'])
            
            for head, num_in_features, head_module in head_list:
                self.train_loss[head][: self.start_epoch] = package['train_loss'][head][: self.start_epoch]
                self.valid_loss[head][: self.start_epoch] = package['valid_loss'][head][: self.start_epoch]
        else:
            self.continue_from = None
            self.model_dir = args.model_dir
            self.start_epoch = 0
    
    def train(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.epochs):
            avg_train_loss = self.run_one_epoch_train(epoch)
            avg_valid_loss = self.run_one_epoch_valid(epoch)
            
            print("[Epoch {}] loss (train): {}, loss (valid): {}".format(epoch+1, avg_train_loss, avg_valid_loss))
            
            model_path = os.path.join(self.model_dir, "epoch{}.pth".format(epoch+1))
            package = {'state_dict': self.model.state_dict(), 'optim_dict': self.optimizer.state_dict(), 'epoch': epoch+1, 'train_loss': self.train_loss, 'valid_loss': self.valid_loss}
            torch.save(package, model_path)
    
    def run_one_epoch_train(self, epoch):
        head_list = self.head_list
        
        self.model.train()
        
        total_loss = 0
        
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
            domain_loss = {}

            print("[Epoch {}] iteration {}/{}, ".format(epoch+1, iteration+1, len(self.train_loader)), end='')
        
            for (head, num_out_features, head_module) in head_list:
                domain_loss[head] = self.criterions[head](estimated_output[head], target[head], target['pointmap'], target['num_object'])
                print("({}): {}, ".format(head, domain_loss[head].item()), end='')
                self.train_loss[head][epoch] = self.train_loss[head][epoch] + domain_loss[head].detach().item()
                loss = loss + self.lambdas[head] * domain_loss[head]

            print("loss: {}".format(loss.item()))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss = total_loss + loss.detach().item()
            
        for (head, num_out_features, head_module) in head_list:
            self.train_loss[head][epoch] = self.train_loss[head][epoch] / len(self.train_loader)
        
        return total_loss / len(self.train_loader)
        
    def run_one_epoch_valid(self, epoch):
        head_list = self.head_list
    
        self.model.eval()
        
        with torch.no_grad():
            for iteration, (valid_images, target) in enumerate(self.valid_loader):
                outputs = self.model(valid_images)
                
                estimated_output = {head: None for (head, num_in_features, head_module) in head_list}
                
                for output in outputs:
                    for head in output.keys():
                        if estimated_output[head] is None:
                            estimated_output[head] = output[head].unsqueeze(dim=0)
                        else:
                            estimated_output[head] = torch.cat((estimated_output[head], output[head].unsqueeze(dim=0)), dim=0)

                for (head, num_out_features, head_module) in head_list:
                    self.valid_loss[head][epoch] = self.valid_loss[head][epoch] + self.criterions[head](estimated_output[head], target[head], target['pointmap'], target['num_object'])

            avg_loss = 0
            
            print("[Epoch {} valid] ".format(epoch+1), end='')
            
            for (head, num_out_features, head_module) in head_list:
                self.valid_loss[head][epoch] = self.valid_loss[head][epoch] / len(self.valid_loader)
                print("({}): {}, ".format(head, self.valid_loss[head][epoch]), end='')
                
                avg_loss = avg_loss + self.lambdas[head] * self.valid_loss[head][epoch]
            
            print("")
                
            return avg_loss
        

class Evaluater(object):
    def __init__(self, data_loader, model, head_list, args):
        self.data_loader = data_loader['test']
        self.model = model
        
        self.R = args.R
        self.head_list = head_list
        self.inv_camera_matrix = get_inv_camera_matrix(args.camera_matrix)
        
        self.out_csv_path = args.out_csv_path
        self.data_frame = pd.DataFrame(columns=['ImageId', 'PredictionString'])
        
        self.out_image_dir = args.out_image_dir
        os.makedirs(self.out_image_dir, exist_ok=True)
        
        
    def eval(self):
        batch_size = self.data_loader.batch_size
        head_list = self.head_list
        R = self.R
        
        self.model.eval()
        
        with torch.no_grad():
            for iteration, (test_images, image_id) in enumerate(self.data_loader):
                outputs = self.model(test_images)
                
                estimated_output = {head: None for (head, num_in_features, head_module) in head_list}
                
                for output in outputs:
                    for head in output.keys():
                        if estimated_output[head] is None:
                            estimated_output[head] = output[head].unsqueeze(dim=0)
                        else:
                            estimated_output[head] = torch.cat((estimated_output[head], output[head].unsqueeze(dim=0)), dim=0)

                for batch_id in range(batch_size):
                    keypoint_list = []
                    prediction_string = []
                    
                    heatmap = estimated_output['heatmap'][-1][batch_id, 0].clone().cpu()
                    heatmap_path = os.path.join(self.out_image_dir, image_id[batch_id] + '_heatmap.png')
                    show_heatmap(heatmap, save_path=heatmap_path)
                    
                    keypoint_map = get_keypoints(heatmap)
                    keypoint_map_path = os.path.join(self.out_image_dir, image_id[batch_id] + '_keypoint.png')
                    show_heatmap(keypoint_map, save_path=keypoint_map_path)
                    
                    for row_id, keypoint_map_row in enumerate(keypoint_map):
                        for column_id, keypoint_map_pixel in enumerate(keypoint_map_row):
                            if keypoint_map[row_id, column_id] > 0.0:
                                keypoint_list.append((row_id, column_id))
                    
                    test_image = test_images[batch_id].clone().cpu().permute(1, 2, 0)
                    image_path = save_path = os.path.join(self.out_image_dir, image_id[batch_id] + '.png')
                    show_image(test_image, save_path=image_path)
                        
                    for row_id, column_id in keypoint_list:
                        depth_map = estimated_output['depth'][-1][batch_id, 0].clone().cpu()
                        estimated_depth = depth_map[row_id, column_id]
                        x_image, y_image = float(row_id*R)*estimated_depth, float(column_id*R)*estimated_depth
                        image_coords = np.array([x_image, y_image, estimated_depth])
                        camera_coords = np.dot(self.inv_camera_matrix, image_coords.T).T
                        confidence = heatmap[row_id, column_id]
                        
                        # TODO: Predict yaw pitch roll
                        prediction = (0.18, 0.0, 3.14159, camera_coords[0], camera_coords[1], camera_coords[2], confidence) # yaw, pitch, roll, x, y, z,confidence
                        prediction_string.append(prediction)
                        
                    prediction_string = [str(prediction).replace('(', '').replace(')', '').replace(',', '') for prediction in prediction_string]
                    prediction_string = ' '.join(prediction_string)
                            
                    data = pd.DataFrame([(image_id[batch_id], prediction_string)], columns=['ImageId', 'PredictionString'])
                    self.data_frame = self.data_frame.append(data)
                        
                break
            
            self.data_frame.to_csv(self.out_csv_path, index=False)
