import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import show_heatmap, show_image, get_keypoints, get_inv_camera_matrix, show_learning_curve, get_angle_shift, get_angle_from_offset

THRESHOLD=0.25

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
        
        if args.coeff_sigma_decay == 0.0 and args.coeff_sigma!=args.coeff_sigma_min:
            raise ValueError("If args.coeff_sigma_decay is 0, then you have to set args.coeff_sigma == args.coeff_sigma_min !")
        
        self.coeff_sigma_decay = args.coeff_sigma_decay
        self.coeff_sigma_min = args.coeff_sigma_min
        
        self.show_target_heatmap = args.show_target_heatmap
        
        if args.continue_from is not None:
            self.model_dir = os.path.dirname(args.continue_from)
            package = torch.load(args.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.start_epoch = package['epoch']
            self.optimizer.load_state_dict(package['optim_dict'])
            self.train_loader.dataset.coeff_sigma = package['coeff_sigma']
            self.valid_loader.dataset.coeff_sigma = package['coeff_sigma']
            
            for head, num_in_features, head_module in head_list:
                self.train_loss[head][: self.start_epoch] = package['train_loss'][head][: self.start_epoch]
                self.valid_loss[head][: self.start_epoch] = package['valid_loss'][head][: self.start_epoch]
        else:
            self.model_dir = args.model_dir
            self.start_epoch = 0
    
    def train(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
        for epoch in range(self.start_epoch, self.epochs):
            avg_train_loss = self.run_one_epoch_train(epoch)
            avg_valid_loss = self.run_one_epoch_valid(epoch)
            
            print("[Epoch {}] loss (train): {}, loss (valid): {}".format(epoch+1, avg_train_loss, avg_valid_loss))
            
            if self.coeff_sigma_decay > 0 and self.train_loader.dataset.coeff_sigma > self.coeff_sigma_min:
                print("Coeff sigma changed {}".format(self.train_loader.dataset.coeff_sigma), end='')
                self.train_loader.dataset.coeff_sigma = self.train_loader.dataset.coeff_sigma - args.coeff_sigma_decay
                self.valid_loader.dataset.coeff_sigma = self.valid_loader.dataset.coeff_sigma + args.coeff_sigma_decay
                print(" -> {}".format(self.train_loader.dataset.coeff_sigma))
                
            model_path = os.path.join(self.model_dir, "epoch{}.pth".format(epoch+1))
            self.save_model(model_path, epoch)
            show_learning_curve(epoch+1, self.head_list, train_loss=self.train_loss, valid_loss=self.valid_loss, save_path=self.model_dir)
    
    def run_one_epoch_train(self, epoch):
        head_list = self.head_list
        
        self.model.train()
        
        total_loss = 0
        
        for iteration, (train_images, target) in enumerate(self.train_loader):
            if torch.cuda.is_available():
                train_images.cuda()
                target = {key: target[key].cuda() for key in target.keys()}
        
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
            
            if iteration < 1 and self.show_target_heatmap:
                for batch_id in range(len(train_images)):
                    heatmap = target['heatmap'][batch_id].detach().clone().cpu()
                    heatmap_path = os.path.join(self.model_dir, 'epoch{}_{}_heatmap(targ).png'.format(epoch+1, batch_id))
                    show_heatmap(heatmap, save_path=heatmap_path)
            
        for (head, num_out_features, head_module) in head_list:
            self.train_loss[head][epoch] = self.train_loss[head][epoch] / len(self.train_loader)
        
        return total_loss / len(self.train_loader)
        
    def run_one_epoch_valid(self, epoch):
        head_list = self.head_list
    
        self.model.eval()
        
        with torch.no_grad():
            for iteration, (valid_images, target) in enumerate(self.valid_loader):
                if torch.cuda.is_available():
                    valid_images.cuda()
                    target = {key: target[key].cuda() for key in target.keys()}
            
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

                if iteration < 1 and 'heatmap' in estimated_output.keys():
                    for batch_id in range(len(valid_images)):
                        heatmap = target['heatmap'][batch_id].detach().clone().cpu()
                        heatmap_path = os.path.join(self.model_dir, 'epoch{}_{}_heatmap(targ).png'.format(epoch+1, batch_id))
                        show_heatmap(heatmap, save_path=heatmap_path)
                        
                        valid_image = valid_images[batch_id].detach().clone().cpu().permute(1, 2, 0)
                        image_path = os.path.join(self.model_dir, 'epoch{}_{}.png'.format(epoch+1, batch_id))
                        show_image(valid_image, save_path=image_path)
                    
                        heatmap = estimated_output['heatmap'][-1][batch_id, 0].detach().clone().cpu()
                        heatmap_max = torch.max(heatmap)
                        heatmap_path = os.path.join(self.model_dir, 'epoch{}_{}_heatmap(est)_{}.png'.format(epoch+1, batch_id, heatmap_max))
                        show_heatmap(heatmap, save_path=heatmap_path)
                        
                        keypoint_map = get_keypoints(heatmap, threshold=THRESHOLD)
                        keypoint_map_path = os.path.join(self.model_dir, 'epoch{}_{}_keypoint(est).png'.format(epoch+1, batch_id))
                        show_heatmap(keypoint_map, save_path=keypoint_map_path)

            avg_loss = 0
            
            print("[Epoch {} valid] ".format(epoch+1), end='')
            
            for (head, num_out_features, head_module) in head_list:
                self.valid_loss[head][epoch] = self.valid_loss[head][epoch] / len(self.valid_loader)
                print("({}): {}, ".format(head, self.valid_loss[head][epoch]), end='')
                
                avg_loss = avg_loss + self.lambdas[head] * self.valid_loss[head][epoch]
            
            print("")
                
            return avg_loss
            
    def save_model(self, model_path, epoch):
        package = {'state_dict': self.model.state_dict(), 'optim_dict': self.optimizer.state_dict(), 'epoch': epoch+1, 'train_loss': self.train_loss, 'valid_loss': self.valid_loss, 'coeff_sigma': self.train_loader.dataset.coeff_sigma}
        torch.save(package, model_path)
        

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
        
        package = torch.load(args.model_path)
        self.model.load_state_dict(package['state_dict'])
        self.data_loader.dataset.coeff_sigma = package['coeff_sigma']
        
        
    def eval(self):
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

                for batch_id in range(len(test_images)):
                    keypoint_list = []
                    prediction_string = []
                    
                    output_keys = estimated_output.keys()
                    
                    if not 'heatmap' in output_keys:
                        break
                    """
                    NOTICE:
                        estimated_output[PROPERTY][-1][BATCH_ID, CHANNEL]:
                            PROPERTY is like 'depth', -1 means last stack, BATCH_ID is number of batch, CHANNEL is #channels of PROPERTY
                    """
                    
                    """ Heatmap estimation """
                    heatmap = estimated_output['heatmap'][-1][batch_id, 0].clone().cpu()
                    heatmap_path = os.path.join(self.out_image_dir, '{}_heatmap.png'.format(image_id[batch_id]))
                    # HERE
                    # show_heatmap(heatmap, save_path=heatmap_path)
                    
                    """ Keypoint estimation """
                    keypoint_map = get_keypoints(heatmap, threshold=THRESHOLD)
                    keypoint_map_path = os.path.join(self.out_image_dir, '{}_keypoint.png'.format(image_id[batch_id]))
                    # HERE
                    # show_heatmap(keypoint_map, save_path=keypoint_map_path)
                    
                    for row_id, keypoint_map_row in enumerate(keypoint_map):
                        for column_id, keypoint_map_pixel in enumerate(keypoint_map_row):
                            if keypoint_map[row_id, column_id] > 0.0:
                                keypoint_list.append((row_id, column_id))
                    
                    """ Property estimation """
                    for row_id, column_id in keypoint_list:
                        confidence = heatmap[row_id, column_id].item()
                        
                        """ X estimation """
                        if 'x' in output_keys:
                            x_map = estimated_output['x'][-1][batch_id, 0].clone().cpu()
                            estimated_x = x_map[row_id, column_id].item()
                        else:
                            estimated_x = 0.0
                        # print("estimated_x is ", estimated_x)
                        
                        """ Y estimation """
                        if 'y' in output_keys:
                            y_map = estimated_output['y'][-1][batch_id, 0].clone().cpu()
                            estimated_y = y_map[row_id, column_id].item()
                        else:
                            estimated_y = 0.0
                        # print("estimated_y is ", estimated_y)
                            
                        """ Depth estimation """
                        if 'depth' in output_keys:
                            depth_map = estimated_output['depth'][-1][batch_id, 0].clone().cpu()
                            estimated_depth = depth_map[row_id, column_id].item()
                        else:
                            depth = 0.0
                        # print("estimated_depth is ", estimated_depth)
                            
                        """
                            x_image, y_image = float(row_id*R)*estimated_depth, float(column_id*R)*estimated_depth
                            image_coords = np.array([x_image, y_image, estimated_depth])
                            camera_coords = np.dot(self.inv_camera_matrix, image_coords.T).T
                        """
                            
                        """ Yaw estimation """
                        if 'yaw' in output_keys:
                            yaw_map = estimated_output['yaw'][-1][batch_id, 0].clone().cpu()
                            estimated_yaw = yaw_map[row_id, column_id].item()
                        else:
                            estimated_yaw = 0.1
                        # print("estimated_yaw is ", estimated_yaw)
                            
                        """ Pitch estimation """
                        if 'categorical_pitch' in output_keys:
                            categorical_pitch_map = estimated_output['categorical_pitch'][-1][batch_id].clone().cpu()
                            estimated_categorical_pitch = categorical_pitch_map[:, :, row_id, column_id]
                            pitch_bin = torch.argmax(estimated_categorical_pitch[:, 0]).int().item()
                            estimated_pitch = estimated_categorical_pitch[pitch_bin, 1].item()
                            get_angle_from_offset(estimated_pitch, category=pitch_bin, num_category=4)
                            # print("pitch_bin is", pitch_bin)
                        else:
                            estimated_pitch = 0.0
                        # print("pitch_offset is", estimated_pitch)
                            
                        """ Roll estimation """
                        if 'shifted_roll' in output_keys:
                            shifted_roll_map = estimated_output['shifted_roll'][-1][batch_id, 0].clone().cpu()
                            estimated_roll = shifted_roll_map[row_id, column_id].item()
                            estimated_roll = get_angle_shift(estimated_roll)
                        else:
                            estimated_roll = 3.14159
                        # print("estimated_roll is ", estimated_roll)
                        
                        prediction = (estimated_yaw, estimated_pitch, estimated_roll, estimated_x, estimated_y, estimated_depth, confidence) # yaw, pitch, roll, x, y, z,confidence
                        prediction_string.append(prediction)
                        
                    prediction_string = [str(prediction).replace('(', '').replace(')', '').replace(',', '') for prediction in prediction_string]
                    prediction_string = ' '.join(prediction_string)
                            
                    data = pd.DataFrame([(image_id[batch_id], prediction_string)], columns=['ImageId', 'PredictionString'])
                    self.data_frame = self.data_frame.append(data)
                    
                    test_image = test_images[batch_id].clone().cpu().permute(1, 2, 0)
                    image_path = os.path.join(self.out_image_dir, image_id[batch_id] + '.png')
                    # HERE
                    # show_image(test_image, save_path=image_path)
    
                # HERE
                # break
            
            self.data_frame.to_csv(self.out_csv_path, index=False)
