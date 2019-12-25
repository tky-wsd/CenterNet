import os
import math
import glob
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.utils.data

from utils import from_prediction_string_to_world_coords, from_camera_coords_to_image_coords, get_front_or_back

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, R, camera_matrix):
        super(TrainDataset, self).__init__()
        
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.R = R
        self.camera_matrix = camera_matrix
        
    def __getitem__(self, idx):
        """
        About important paramters in this code
            target_pointmap:
            target_heatmap:
            target_local_offset:
            target_depth:
            target_pitch (2, H//R, W//R): 0 or 1, sin()
        """
        R = self.R
    
        image_path = os.path.join(self.image_dir, self.data['ImageId'][idx] + '.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        H, W, N = image.size() # H, W, 3
        image = image.permute(2, 0, 1).float()
        image = 2*(image/255)-1

        car_id, euler_angle, world_coords = from_prediction_string_to_world_coords(self.data['PredictionString'][idx])
        image_coords = from_camera_coords_to_image_coords(world_coords, self.camera_matrix)
        image_coords = torch.from_numpy(image_coords).float()
        
        target_pointmap = torch.zeros(H//R, W//R, dtype=torch.float)
        target_heatmap = None
        target_local_offset = torch.zeros(2, H//R, W//R, dtype=torch.float)
        target_depth = torch.zeros(H//R, W//R, dtype=torch.float)
        target_yaw = torch.zeros(H//R, W//R, dtype=torch.float)
        target_pitch = torch.zeros(2, H//R, W//R, dtype=torch.float)
        target_roll = torch.zeros(H//R, W//R, dtype=torch.float)
        h = torch.arange(0, H//R, 1, dtype=torch.float).unsqueeze(dim=1).expand(-1, W//R)
        w = torch.arange(0, W//R, 1, dtype=torch.float).unsqueeze(dim=0).expand(H//R, -1)
        p_h = image_coords[:, 1]
        p_w = image_coords[:, 0]
        
        num_object = torch.Tensor([len(car_id)])
        
        for object_id in range(int(num_object.item())):
            point = ((p_h[object_id]/R).int(), (p_w[object_id]/R).int())
            
            if point[0] < 0 or H//R - 1 < point[0] or point[1] < 0 or W//R - 1 < point[1] :
                continue
            target_pointmap[point[0], point[1]] = 1
            target_local_offset[0][point[0], point[1]] = p_h[object_id]/R - point[0].float()
            target_local_offset[1][point[0], point[1]] = p_w[object_id]/R - point[1].float()
            target_depth[point[0], point[1]] = image_coords[object_id, 2]
            target_yaw[point[0], point[1]] = math.sin(euler_angle[object_id][0])
            target_pitch[0][point[0], point[1]] = float(get_front_or_back(euler_angle[object_id][1]))
            target_pitch[1][point[0], point[1]] = math.cos(euler_angle[object_id][1])
            target_roll[point[0], point[1]] = math.sin(euler_angle[object_id][2])
            
            sigma = 100 / target_depth[point[0], point[1]]
            exponent = - ((h-p_h[object_id]/R)**2 + (w-p_w[object_id]/R)**2)/(2 * torch.pow(sigma, 2))
            heatmap = torch.exp(exponent)
                
            if target_heatmap is None:
                target_heatmap = heatmap.unsqueeze(dim=0)
            else:
                target_heatmap = torch.cat((target_heatmap, heatmap.unsqueeze(dim=0)), dim=0)
                

        target_heatmap, _ = torch.max(target_heatmap, dim=0)
        
        if torch.cuda.is_available():
            image = image.cuda().contiguous()
            target = {'num_object': num_object.cuda(), 'pointmap': target_pointmap.cuda(), 'heatmap': target_heatmap.cuda(), 'local_offset': target_local_offset.cuda(), 'depth': target_depth.cuda(), 'yaw': target_yaw.cuda(), 'pitch': target_pitch.cuda(), 'roll': target_roll.cuda()}
        else:
            image = image.contiguous()
            target = {'num_object': num_object, 'pointmap': target_pointmap, 'heatmap': target_heatmap, 'local_offset': target_local_offset, 'depth': target_depth, 'yaw': target_yaw, 'pitch': target_pitch, 'roll': target_roll}
        
        return image, target
                
    def __len__(self):
        return len(self.data)
        
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, R, camera_matrix):
        super(TestDataset, self).__init__()
        
        self.data = []
        
        data = [os.path.splitext(os.path.basename(path))[0] for path in glob.glob(os.path.join(image_dir, "*.jpg"))]
        data = pd.DataFrame(data, columns=['ImageId'])
        data['PredictionString'] = ''
        
        self.data = data
        
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.R = R
        self.camera_matrix = camera_matrix
            
    def __getitem__(self, idx):
        R = self.R
        
        image_id = self.data['ImageId'][idx]
    
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        H, W, N = image.size() # H, W, 3
        image = image.permute(2, 0, 1).float()
        image = 2*(image/255)-1
        
        if torch.cuda.is_available():
            image = image.cuda().contiguous()
        else:
            image = image.contiguous()
        
        return image, image_id
                    
    def __len__(self):
        return len(self.data)
            
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    f_x, f_y = 2304.5479, 2305.8757
    c_x, c_y = 1686.2379, 1354.9849
    
    camera_matrix = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    
    dataset = TrainDataset(csv_path="../data/train.csv", image_dir="../data/train_images", R=4, camera_matrix=camera_matrix)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for data in data_loader:
        print(data)
        
        break
