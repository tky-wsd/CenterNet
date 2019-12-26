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

def get_front_or_back(angle):
    """
    Args:
        angle:
    Returns:
        is_front: if abs(angle) < math.pi / 2, returns 0
    """
    if abs(angle) < math.pi / 2:
        is_front = 1
    else:
        is_front = 0
    return is_front
        
def get_keypoints(heatmap, threshold=0.6):
    """
    Args:
        heatmap (H//R, W//R):
        
    """
    if threshold < 0.5:
        pass
        # raise ValueError("Parametes threshold must be grater than 0.5.")
    H, W = heatmap.size()
    
    mask = heatmap // threshold
    heatmap = mask * heatmap
    
    estimated_key_point = np.zeros((H, W))
    
    for p_h in range(1, H-1):
        for p_w in range(1, W-1):
            if mask[p_h, p_w] > 0.0:
                if is_higher_than_around(heatmap[p_h-1: p_h+2, p_w-1: p_w+2]):
                    estimated_key_point[p_h, p_w] = 1.0
                    
    return estimated_key_point
                    
def is_higher_than_around(map_3x3):
    """
    Args:
        map_3x3 (3, 3)
    """
    
    for row in range(3):
        for column in range(3):
            if map_3x3[2, 2] < map_3x3[row, column]:
                return False
                
    return True
        
                
