import os
import math
import glob
import pandas as pd
import imageio
import torch
import torch.utils.data

from utils import from_prediction_string_to_world_coords, from_camera_coords_to_image_coords, get_angle_category, get_angle_bin_offset, get_angle_shift

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, R, camera_matrix, cut_upper=False, coeff_sigma=100.0):
        super(TrainDataset, self).__init__()
        
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.R = R
        self.camera_matrix = camera_matrix
        
        self.cut_upper = cut_upper
        self.coeff_sigma = coeff_sigma
        
    def __getitem__(self, idx):
        """
        About important paramters in this code
            target_pointmap:
            target_heatmap:
            target_local_offset:
            target_depth:
            target_pitch (4, 2, H//R, W//R):
        """
        R = self.R
    
        image_path = os.path.join(self.image_dir, self.data['ImageId'][idx] + '.jpg')
        image = imageio.imread(image_path)
        image = torch.from_numpy(image)
        
        H, W, _ = image.size()
        
        image = image.permute(2, 0, 1).float() # H, W, 3 -> 3, H, W
        image = 2*(image/255)-1

        car_id, euler_angle, world_coords = from_prediction_string_to_world_coords(self.data['PredictionString'][idx])
        image_coords = from_camera_coords_to_image_coords(world_coords, self.camera_matrix)
        image_coords = torch.from_numpy(image_coords).float()
        
        if self.cut_upper:
            H = H//2
            image = image[:, :H, :]
            image_coords[:, 1] = image_coords[:, 1] / 2.0
        
        target_pointmap = torch.zeros(H//R, W//R, dtype=torch.float)
        
        target_heatmap = None
        target_x = torch.zeros(H//R, W//R, dtype=torch.float)
        target_y = torch.zeros(H//R, W//R, dtype=torch.float)
        target_local_offset = torch.zeros(2, H//R, W//R, dtype=torch.float)
        target_depth = torch.zeros(H//R, W//R, dtype=torch.float)
        target_yaw = torch.zeros(H//R, W//R, dtype=torch.float)
        target_categorical_pitch = torch.zeros(4, 2, H//R, W//R, dtype=torch.float)
        target_roll = torch.zeros(H//R, W//R, dtype=torch.float)
        target_shifted_roll = torch.zeros(H//R, W//R, dtype=torch.float)
        h = torch.arange(0, H//R, 1, dtype=torch.float).unsqueeze(dim=1).expand(-1, W//R)
        w = torch.arange(0, W//R, 1, dtype=torch.float).unsqueeze(dim=0).expand(H//R, -1)
        p_h = image_coords[:, 1]
        p_w = image_coords[:, 0]
        
        num_object = torch.Tensor([len(car_id)])
        
        for object_id in range(int(num_object.item())):
            point = ((p_h[object_id]/R).int(), (p_w[object_id]//R).int())
            
            if point[0] < 0 or H//R - 1 < point[0] or point[1] < 0 or W//R - 1 < point[1] :
                continue
            target_pointmap[point[0], point[1]] = 1
            target_local_offset[0][point[0], point[1]] = p_h[object_id]/R - point[0].float()
            target_local_offset[1][point[0], point[1]] = p_w[object_id]/R - point[1].float()
            target_x[point[0], point[1]] = torch.Tensor([world_coords[object_id][0]])
            target_y[point[0], point[1]] = torch.Tensor([world_coords[object_id][1]])
            target_depth[point[0], point[1]] = image_coords[object_id, 2]
            target_yaw[point[0], point[1]] = torch.Tensor([euler_angle[object_id][0]])
            target_pitch = torch.Tensor([euler_angle[object_id][1]])
            target_roll[point[0], point[1]] = torch.Tensor([euler_angle[object_id][2]])
            
            pitch_bin = get_angle_category(euler_angle[object_id][1], num_category=4)
            target_categorical_pitch[pitch_bin][0][point[0], point[1]] = 1
            target_categorical_pitch[pitch_bin][1][point[0], point[1]] = get_angle_bin_offset(torch.Tensor([euler_angle[object_id][1]]), category=pitch_bin, num_category=4)
            target_shifted_roll[point[0], point[1]] = get_angle_shift(torch.Tensor([euler_angle[object_id][2]]))
            
            sigma = self.coeff_sigma * 1.1**(-target_depth[point[0], point[1]]/10) # 10 * {1.1^(-d/10)}
            exponent = - ((h-p_h[object_id]/R)**2 + (w-p_w[object_id]/R)**2)/(2 * torch.pow(sigma, 2))
            heatmap = torch.exp(exponent)
                
            if target_heatmap is None:
                target_heatmap = heatmap.unsqueeze(dim=0)
            else:
                target_heatmap = torch.cat((target_heatmap, heatmap.unsqueeze(dim=0)), dim=0)

        target_heatmap, _ = torch.max(target_heatmap, dim=0)
        
        image = image.contiguous()
        target = {
            'num_object': num_object, 'pointmap': target_pointmap,
            'heatmap': target_heatmap, 'local_offset': target_local_offset,
            'x': target_x, 'y': target_y,
            'depth': target_depth,
            'yaw': target_yaw, 'pitch': target_pitch, 'roll': target_roll,
            'categorical_pitch': target_categorical_pitch, 'shifted_roll': target_shifted_roll
        }
       
        return image, target
                
    def __len__(self):
        return len(self.data)
        
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, R, camera_matrix):
        super(TestDataset, self).__init__()
        
        self.data = []
        
        data_frame = pd.read_csv(csv_path)
        self.image_id_list = [image_id for image_id in data_frame['ImageId']]
        
        self.image_dir = image_dir
        self.R = R
        self.camera_matrix = camera_matrix
            
    def __getitem__(self, idx):
        image_id = self.image_id_list[idx]
    
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        image = imageio.imread(image_path)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float()
        image = 2*(image/255)-1
        
        image = image.contiguous()
        
        return image, image_id
                    
    def __len__(self):
        return len(self.image_id_list)
            
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
