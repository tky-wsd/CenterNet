import numpy as np


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
