import os
import math
import numpy as np
import matplotlib.pyplot as plt

def from_prediction_string_to_world_coords(prediction_string):
    """
    Args:
        prediction_string: given PredictionString
    Returns:
        car_id (batch_size, ): ID of car
        euler_angle (batch_size, 3): Euler angle (yaw, pitch, roll)
        camera_coords (batch_size, 3): (x, y, z) in camera coordinates
    """
    camera_coords = [value for value in prediction_string.split(' ')]
    camera_coords = np.array(camera_coords).reshape(-1, 7)
    car_id = camera_coords[:, 0]
    euler_angle = camera_coords[:, 1: 4].astype(dtype=np.float32)
    camera_coords = camera_coords[:, 4:].astype(dtype=np.float32)

    return car_id, euler_angle, camera_coords

def from_camera_coords_to_image_coords(camera_coords, camera_matrix):
    """
    Args:
        camera_coords (batch_size, 3):
    Returns:
         image_coords (batch_size, 3):
    """
    image_coords = np.dot(camera_matrix, camera_coords.T).T # image_coords (batch_size, 3)
    image_coords[:, 0] =  image_coords[:, 0] / image_coords[:, 2]
    image_coords[:, 1] =  image_coords[:, 1] / image_coords[:, 2]

    return image_coords
    
def show_heatmap(heatmap, save_path=None):
    """
    Args:
        heatmap (H//R, W//R)
        image (H, W)
    
    """
    plt.figure()
    plt.imshow(heatmap)
    
    if save_path is not None:
        plt.savefig(save_path)
            
def show_image(image, save_path=None):
    plt.figure()
    image = ((image+1)/2*255).int()
    plt.imshow(image)
    
    if save_path is not None:
        plt.savefig(save_path)
        
def show_learning_curve(epoch, head_list, train_loss, valid_loss=None, save_path="/tmp"):
    x = np.arange(1, epoch+1)
    
    for head, _, _ in head_list:
        plt.figure()
        y = train_loss[head][:epoch].numpy()
        plt.plot(x, y)
        
        if valid_loss is not None:
            y = valid_loss[head][:epoch].numpy()
            plt.plot(x, y)
        
        plt.savefig(os.path.join(save_path, "{}.png".format(head)))
    

def get_angle_category(angle, num_category=4):
    """
    Args:
        angle:
    Returns:
        bin_id
    """
    if - math.pi/4 < angle and angle <= math.pi/4:
        return 0
    if math.pi/4 < angle and angle <= 3*math.pi/4:
        return 1
    if (3*math.pi/4 < angle and angle <= math.pi) or (-math.pi <= angle and angle <= -3*math.pi/4):
        return 2
    if -3*math.pi/4 < angle and angle < -math.pi/4:
        return 3
    
    return 0
    
def get_angle_bin_offset(angle, category, num_category=4):
    """
    Returns:
        angle_bin_offset: Offset from; 0, math.pi/2, math.pi, -math.pi/2
    """
    angle_center = [0, math.pi/2, math.pi, -math.pi/2]
    
    if category == 2 and angle < 0:
        angle = angle + 2*math.pi
    
    angle_bin_offset = angle - angle_center[category]
        
    return angle_bin_offset
    
def get_angle_from_offset(angle_bin_offset, category, num_category=4):
    """
    From category & offset, get angle
    """
    angle_center = [0, math.pi/2, math.pi, -math.pi/2]
    
    angle = angle_center[category] + angle_bin_offset
    
    if category == 2 and angle > math.pi:
        angle = angle - 2 * math.pi
    
    return angle
    
    
def get_angle_shift(angle):
    angle = angle + math.pi # [-math.pi, math.pi) -> [0, 2*math.pi)
    
    if angle > math.pi:
        angle = angle - 2 * math.pi # [0, 2*math.pi) -> [-math.pi, math.pi)
        
    return angle
    
        
def get_keypoints(heatmap, threshold=0.6):
    """
    Args:
        heatmap (H//R, W//R):
        
    """
    H, W = heatmap.size()
    
    mask = heatmap // threshold
    heatmap = mask * heatmap
    
    estimated_key_point = np.zeros((H, W))
    
    for p_h in range(2, H-2):
        for p_w in range(2, W-2):
            if mask[p_h, p_w] > 0.0 and is_higher_than_around(heatmap[p_h-2: p_h+3, p_w-2: p_w+3]):
                estimated_key_point[p_h, p_w] = 1.0
                    
    return estimated_key_point
                    
def is_higher_than_around(map_5x5):
    """
    Args:
        map_5x5 (5, 5)
    """
    
    for row in range(5):
        for column in range(5):
            if map_5x5[2, 2] < map_5x5[row, column]:
                return False
                
    return True
        
                
def get_inv_camera_matrix(camera_matrix):
    inv_camera_matrix = np.eye(3)
    inv_camera_matrix[0,0] = 1.0 / camera_matrix[0,0]
    inv_camera_matrix[1,1] = 1.0 / camera_matrix[1,1]
    inv_camera_matrix[0,2]  = - camera_matrix[0,2] / camera_matrix[0,0]
    inv_camera_matrix[1,2]  = - camera_matrix[1,2] / camera_matrix[1,1]
    
    return inv_camera_matrix
