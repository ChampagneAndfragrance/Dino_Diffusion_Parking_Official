import json
import os
import carla
import torch.utils.data
import numpy as np
import torchvision.transforms
import yaml

from PIL import Image
from loguru import logger
#import matplotlib.pyplot as plt
import math

def rotation_to_matrix(rotation: carla.Rotation):
    """Convert CARLA Rotation (degrees) to a 3×3 rotation matrix."""
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Roll (X) -> Pitch (Y) -> Yaw (Z)
    return Rz @ Ry @ Rx

def rotate_vector(vec: carla.Vector3D, rotation: carla.Rotation):
    """Rotate a vector using carla.Rotation."""
    R = rotation_to_matrix(rotation)
    v = np.array([vec.x, vec.y, vec.z])
    v_rot = R @ v
    return carla.Vector3D(*v_rot)

def get_inverse_transform(transform: carla.Transform):
    """Manually compute inverse of a Transform (rotation + translation)."""
    inv_rot = carla.Rotation(
        pitch=-transform.rotation.pitch,
        yaw=-transform.rotation.yaw,
        roll=-transform.rotation.roll
    )

    R = rotation_to_matrix(inv_rot)
    t = np.array([-transform.location.x, -transform.location.y, -transform.location.z])
    t_rot = R @ t
    inv_loc = carla.Location(x=t_rot[0], y=t_rot[1], z=t_rot[2])

    return carla.Transform(inv_loc, inv_rot)

def world_vector_to_ego(vector: carla.Vector3D, ego_transform: carla.Transform):
    """Convert a world-frame vector (e.g., speed or acceleration) to ego-frame."""
    inv_transform = get_inverse_transform(ego_transform)
    return rotate_vector(vector, inv_transform.rotation)

def world_location_to_ego(location: carla.Location, ego_transform: carla.Transform):
    """Convert a world-frame location to ego-frame location."""
    inv_transform = get_inverse_transform(ego_transform)
    return inv_transform.transform(location)

def convert_slot_coord(ego_trans, target_point):
    """
    Convert target parking slot from world frame into self_veh frame
    :param ego_trans: veh2world transform
    :param target_point: target parking slot in world frame [x, y, yaw]
    :return: target parking slot in veh frame [x, y, yaw]
    """

    target_point_self_veh = convert_veh_coord(target_point[0], target_point[1], 1.0, ego_trans)

    yaw_diff = target_point[2] - ego_trans.rotation.yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    target_point = [target_point_self_veh[0], target_point_self_veh[1], (yaw_diff)]

    return target_point

def convert_veh_in_slot_frame(ego_trans, target_point):
    """
    Convert vehicle position from world frame into parking slot frame.
    :param slot_trans: slot pose in world frame (Transform)
    :param veh_trans: vehicle pose in world frame (Transform)
    :return: [x, y, yaw] in slot frame
    """

    # Convert vehicle position to slot frame
    # veh_in_slot = convert_veh_coord(
    #     ego_trans.location.x,
    #     ego_trans.location.y,
    #     1.0,
    #     target_point  # <-- here slot_trans is used as "reference" frame
    # )

    # Compute yaw difference (vehicle relative to slot)
    yaw_diff = ego_trans.rotation.yaw - target_point[2]
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    return [ego_trans.location.x - target_point[0], ego_trans.location.y - target_point[1], (yaw_diff)]

def convert_veh_coord(x, y, z, ego_trans):
    """
    Convert coordinate (x,y,z) in world frame into self-veh frame
    :param x:
    :param y:
    :param z:
    :param ego_trans: veh2world transform
    :return: coordinate in self-veh frame
    """

    world2veh = np.array(ego_trans.get_inverse_matrix())
    target_array = np.array([x, y, z, 1.0], dtype=float)
    target_point_self_veh = world2veh @ target_array
    return target_point_self_veh


def scale_and_crop_image(image, scale=1.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array
    :param image: original image
    :param scale: scale factor
    :param crop: crop size
    :return: cropped image
    """

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), resample=Image.NEAREST)
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop].copy()
    return cropped_image


def tokenize_control(throttle, brake, steer, reverse, token_nums=200):
    """
    Tokenize control signal
    :param throttle: [0,1]
    :param brake: [0,1]
    :param steer: [-1,1]
    :param reverse: {0,1}
    :param token_nums: size of token
    :return: tokenized control range [0, token_nums-4]
    """

    valid_token = token_nums - 4
    half_token = valid_token / 2

    if brake != 0.0:
        throttle_brake_token = int(half_token * (-brake + 1))
    else:
        throttle_brake_token = int(half_token * (throttle + 1))
    steer_token = int((steer + 1) * half_token)
    reverse_token = int(reverse * valid_token)
    return [throttle_brake_token, steer_token, reverse_token]


def detokenize_control(token_list, token_nums=204):
    """
    Detokenize control signals
    :param token_list: [throttle_brake, steer, reverse]
    :param token_nums: size of token number
    :return: control signal values
    """

    valid_token = token_nums - 4
    half_token = float(valid_token / 2)

    if token_list[0] > half_token:
        throttle = token_list[0] / half_token - 1
        brake = 0.0
    else:
        throttle = 0.0
        brake = -(token_list[0] / half_token - 1)

    steer = (token_list[1] / half_token) - 1
    reverse = (True if token_list[2] > half_token else False)

    return [throttle, brake, steer, reverse]


def tokenize_waypoint(x, y, yaw, token_nums=204):
    """
    Tokenize control signal
    :param x: [-1.5, 1.5]
    :param y: [-2, 2]
    :param yaw: [-20, 20]
    :param token_nums: size of token
    :return: tokenized x, y, yaw
    """
    token_nums = token_nums -4
    # Helper function to tokenize a single value
    def tokenize_single_value(value, min_value, max_value):
        # Normalize to [0, 1]
        normalized_value = (value - min_value) / (max_value - min_value)
        # Scale to [0, token_nums]
        tokenized_value = normalized_value * token_nums
        # Ensure the tokenized value is within [0, token_nums]
        tokenized_value = max(0, min(token_nums, tokenized_value))
        return int(tokenized_value)

    # Tokenize each parameter
    x_token = tokenize_single_value(x, -6, 6)
    y_token = tokenize_single_value(y, -6, 6)
    yaw_token = tokenize_single_value(yaw, -40, 40)

    return  [x_token, y_token, yaw_token]



def detokenize_waypoint(token_list, token_nums=204):
    """
    Detokenize waypoint values
    :param token_list: [x_token, y_token, yaw_token]
    :param token_nums: size of token number
    :return: original x, y, yaw values
    """
    token_nums -= 4  # Adjusting for the valid range of tokens

    # Helper function to detokenize a single value
    def detokenize_single_value(token, min_value, max_value):
        # Scale token from [0, token_nums] to [0, 1]
        normalized_value = token / token_nums
        # Scale and shift the normalized value to its original range
        original_value = (normalized_value * (max_value - min_value)) + min_value
        return original_value

    # Detokenize each parameter
    x = detokenize_single_value(token_list[0], -6, 6)
    y = detokenize_single_value(token_list[1], -6, 6)
    yaw = detokenize_single_value(token_list[2], -40, 40)

    return [x, y, yaw]

def get_depth(depth_image_path, crop):
    """
    Convert carla RGB depth image into single channel depth in meters
    :param depth_image_path: carla depth image in RGB format
    :param crop: crop size
    :return: numpy array of depth image in meters
    """
    depth_image = Image.open(depth_image_path).convert('RGB')

    data = np.array(scale_and_crop_image(depth_image, scale=1.0, crop=crop))

    data = data.astype(np.float32)

    normalized = np.dot(data, [1.0, 256.0, 65536.0])
    normalized /= (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized

    return torch.from_numpy(in_meters).unsqueeze(0)


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    update_intrinsic = intrinsics.clone()

    update_intrinsic[0, 0] *= scale_width
    update_intrinsic[0, 2] *= scale_width
    update_intrinsic[1, 1] *= scale_height
    update_intrinsic[1, 2] *= scale_height

    update_intrinsic[0, 2] -= left_crop
    update_intrinsic[1, 2] -= top_crop

    return update_intrinsic


def add_raw_control(data, throttle_brake, steer, reverse):
    if data['Brake'] != 0.0:
        throttle_brake.append(-data['Brake'])
    else:
        throttle_brake.append(data['Throttle'])
    steer.append(data['Steer'])
    reverse.append(int(data['Reverse']))

def compute_shaped_rtg_with_terminal_bonus(
    target_points,
    x_thresh=1.0,
    y_thresh=0.6,
    theta_thresh=10.0,
    step_goal_bonus=5.0,
    terminal_bonus=300.0,
    w_x=1.0,
    w_y=1.0,
    w_theta=0.1,
):
    """
    target_points: list or np.array of shape (T, 3)
    Returns:
        rewards: (T,)
        return_to_go: (T,)
    """
    T = len(target_points)
    target_points = np.array(target_points)
    final_x, final_y, final_theta = target_points[-1]

    def angle_diff(a, b):
        diff = a - b
        return (diff + 180) % 360 - 180

    rewards = []
    for i in range(T):
        x, y, theta = target_points[i]
        dx = abs(x - final_x)
        dy = abs(y - final_y)
        dtheta = abs(angle_diff(theta, final_theta))

        # reward shaping
        reward = - (w_x * dx + w_y * dy + w_theta * dtheta)

        # goal proximity bonus
        in_goal_range = dx <= x_thresh and dy <= y_thresh and dtheta <= theta_thresh
        if in_goal_range:
            if i == T - 1:
                reward += terminal_bonus
            else:
                reward += step_goal_bonus

        rewards.append(reward)

    # Compute return-to-go
    rtg = np.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return += rewards[t]
        rtg[t] = running_return

    return np.array(rewards), rtg



def make_rtg_windowed_array(rtg, window_size=4):
    rtg = np.asarray(rtg)
    T = len(rtg)
    
    # Pad the end with the last value
    padded = np.concatenate([rtg, np.full((window_size - 1,), rtg[-1])])
    
    # Create the (T, window_size) array
    rtg_windowed = np.stack([padded[i:i + window_size] for i in range(T)], axis=0)
    return rtg_windowed  # shape: (220, 4)


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train, config):
        super(CarlaDataset, self).__init__()
        self.cfg = config

        self.BOS_token = self.cfg.token_nums - 3
        self.EOS_token = self.BOS_token + 1
        self.PAD_token = self.EOS_token + 1

        self.root_dir = root_dir
        self.is_train = is_train

        # camera configs
        self.image_crop = self.cfg.image_crop
        self.intrinsic = None
        self.veh2cam_dict = {}
        self.extrinsic = None
        self.image_process = ProcessImage(self.image_crop)
        self.semantic_process = ProcessSemantic(self.cfg)

        self.init_camera_config()

        # data
        self.task_idx = []
        self.ego_trans = []

        self.front = []
        self.front_left = []
        self.front_right = []
        self.back = []
        self.back_left = []
        self.back_right = []

        self.front_depth = []
        self.front_left_depth = []
        self.front_right_depth = []
        self.back_depth = []
        self.back_left_depth = []
        self.back_right_depth = []



        self.control = []

        self.speed = []
        self.acc_x = []
        self.acc_y = []
        self.ego_motion_seq = []

        self.throttle_brake = []
        self.steer = []
        self.reverse = []

        self.target_point = []
        self.target_point_seq = []
        self.acc_return = []

        self.topdown = []

        self.waypoint = []
        self.delta_x_values = []
        self.delta_y_values = []
        self.delta_yaw_values = []

        self.plot_x = []
        self.plot_y = []
        self.plot_yaw = []
        self.get_data()

    def init_camera_config(self):
        cam_config = {'width': 400, 'height': 300, 'fov': 100}
        with open('./config/sensor_specs.yaml', 'r') as file:
            data = yaml.safe_load(file)
            cam_specs = data["cam_specs"]
        # intrinsic
        w = cam_config['width']
        h = cam_config['height']
        fov = cam_config['fov']
        f = w / (2 * np.tan(fov * np.pi / 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic_original = np.array([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ], dtype=float)
        self.intrinsic = update_intrinsics(
            torch.from_numpy(intrinsic_original).float(),
            (h - self.image_crop) / 2,
            (w - self.image_crop) / 2,
            scale_width=1,
            scale_height=1
        )
        self.intrinsic = self.intrinsic.unsqueeze(0).expand(6, 3, 3)

        # extrinsic
        cam2pixel = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        for cam_id, cam_spec in cam_specs.items():
            cam2veh = carla.Transform(carla.Location(x=cam_spec['x'], y=cam_spec['y'], z=cam_spec['z']),
                                      carla.Rotation(yaw=cam_spec['yaw'], pitch=cam_spec['pitch'],
                                                     roll=cam_spec['roll']))
            veh2cam = cam2pixel @ np.array(cam2veh.get_inverse_matrix())
            self.veh2cam_dict[cam_id] = veh2cam
        front_to_ego = torch.from_numpy(self.veh2cam_dict['camera_front']).float().unsqueeze(0)
        front_left_to_ego = torch.from_numpy(self.veh2cam_dict['camera_front_left']).float().unsqueeze(0)
        front_right_to_ego = torch.from_numpy(self.veh2cam_dict['camera_front_right']).float().unsqueeze(0)
        back_to_ego = torch.from_numpy(self.veh2cam_dict['camera_back']).float().unsqueeze(0)
        back_left_to_ego = torch.from_numpy(self.veh2cam_dict['camera_back_left']).float().unsqueeze(0)
        back_right_to_ego = torch.from_numpy(self.veh2cam_dict['camera_back_right']).float().unsqueeze(0)
        self.extrinsic = torch.cat([front_to_ego, front_left_to_ego, front_right_to_ego, back_to_ego,back_left_to_ego,back_right_to_ego], dim=0)

    def get_data(self):
        val_towns = self.cfg.validation_map
        train_towns = self.cfg.training_map
        train_data = os.path.join(self.root_dir, train_towns)
        val_data = os.path.join(self.root_dir, val_towns)

        town_dir = train_data if self.is_train == 1 else val_data

        # collect all parking data tasks
        root_dirs = os.listdir(town_dir)
        all_tasks = []
        for root_dir in root_dirs:
            root_path = os.path.join(town_dir, root_dir)
            for task_dir in os.listdir(root_path):
                task_path = os.path.join(root_path, task_dir)
                all_tasks.append(task_path)

        for task_id, task_path in enumerate(all_tasks):
            pose_episode = []
            total_frames = len(os.listdir(task_path + "/measurements/"))
            for frame in range(self.cfg.hist_frame_nums, total_frames - self.cfg.future_frame_nums):
                # collect data at current frame
                # image
                filename = f"{str(frame).zfill(4)}.png"
                self.front.append(task_path + "/camera_front/" + filename)
                self.front_left.append(task_path + "/camera_front_left/" + filename)
                self.front_right.append(task_path + "/camera_front_right/" + filename)
                self.back.append(task_path + "/camera_back/" + filename)
                self.back_left.append(task_path + "/camera_back_left/" + filename)
                self.back_right.append(task_path + "/camera_back_right/" + filename)

                # depth
                self.front_depth.append(task_path + "/depth_front/" + filename)
                self.front_left_depth.append(task_path + "/depth_front_left/" + filename)
                self.front_right_depth.append(task_path + "/depth_front_right/" + filename)
                self.back_depth.append(task_path + "/depth_back/" + filename)
                self.back_left_depth.append(task_path + "/depth_back_left/" + filename)
                self.back_right_depth.append(task_path + "/depth_back_right/" + filename)

                # BEV Semantic
                self.topdown.append(task_path + "/topdown/encoded_" + filename)

                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)

                # ego position
                ego_trans = carla.Transform(carla.Location(x=data['x'], y=data['y'], z=data['z']),
                                            carla.Rotation(yaw=data['yaw'], pitch=data['pitch'], roll=data['roll']))
                # # motion
                # speed_world = carla.Vector3D(x=data['speed_x'], y=data['speed_y'], z=data['speed_z'])
                # accel_world = carla.Vector3D(x=data['acc_x'], y=data['acc_y'], z=data['acc_z'])

                # # apply transformation
                # speed_ego = world_vector_to_ego(speed_world, ego_trans)
                # accel_ego = world_vector_to_ego(accel_world, ego_trans)
                self.speed.append(data['speed'])

                self.acc_x.append(data['acc_x'])
                self.acc_y.append(data['acc_y'])

                # construct ego motion seq
                ego_motion_seq = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                        ego_motion_seq.append([data['speed'], data['acc_x'], data['acc_y']])
                self.ego_motion_seq.append(ego_motion_seq)

                # control
                controls = []
                throttle_brakes = []
                steers = []
                reverse = []

                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + 1 + i).zfill(4)}.json", "r") as read_file:
                        data = json.load(read_file)
                    controls.append(
                        tokenize_control(data['Throttle'], data["Brake"], data["Steer"], data["Reverse"], self.cfg.token_nums))
                    add_raw_control(data, throttle_brakes, steers, reverse)

                controls = [item for sublist in controls for item in sublist]
                controls.insert(0, self.BOS_token)
                controls.append(self.EOS_token)
                controls.append(self.PAD_token)
                self.control.append(controls)

                self.throttle_brake.append(throttle_brakes)
                self.steer.append(steers)
                self.reverse.append(reverse)

                # waypoint
                with open(task_path + f"/measurements/{str(frame).zfill(4)}.json", "r") as read_file:
                    data = json.load(read_file)
                    cur_x = data['x']
                    cur_y = data['y']
                    cur_yaw = data['yaw']
                waypoints = []
                xs = []
                ys = []
                yaws = []


                for i in range(self.cfg.future_frame_nums):
                    file_path = task_path + f"/measurements/{str(frame + 1 + 10*i ).zfill(4)}.json"
                    if not os.path.exists(file_path):
                        # If the file doesn't exist, use the last frame
                        file_path = task_path + f"/measurements/{str(frame).zfill(4)}.json"
                    with open(file_path, "r") as read_file:
                        data = json.load(read_file)
                        egocentric_waypoint = convert_slot_coord(ego_trans,[data['x'],data['y'],data['yaw']])
                        delta_x = egocentric_waypoint[0]
                        delta_y = egocentric_waypoint[1]
                        delta_yaw = egocentric_waypoint[2]

                        self.plot_x.append(delta_x)
                        self.plot_y.append(delta_y)
                        self.plot_yaw.append(delta_yaw)
                    waypoints.append(
                        tokenize_waypoint(delta_x, delta_y, delta_yaw, self.cfg.token_nums))
                    xs.append(delta_x)
                    ys.append(delta_y)
                    yaws.append(delta_yaw)

                    # add_raw_control(data, throttle_brakes, steers, reverse)

                waypoints = [item for sublist in waypoints for item in sublist]
                waypoints.insert(0, self.BOS_token)
                waypoints.append(self.EOS_token)
                waypoints.append(self.PAD_token)
                self.waypoint.append(waypoints)

                self.delta_x_values.append(xs)
                self.delta_y_values.append(ys)
                self.delta_yaw_values.append(yaws)


                # target point
                with open(task_path + f"/parking_goal/0001.json", "r") as read_file:
                    data = json.load(read_file)
                parking_goal = [data['x'], data['y'], (data['yaw'])]
                parking_goal = convert_slot_coord(ego_trans, parking_goal)
                self.target_point.append(parking_goal)
                pose_episode.append(parking_goal)
                

                target_point_seq = []
                for i in range(self.cfg.future_frame_nums):
                    with open(task_path + f"/measurements/{str(frame + i).zfill(4)}.json", "r") as m_file, open(task_path + f"/parking_goal/0001.json", "r") as p_file:
                        m_data, p_data = json.load(m_file), json.load(p_file)
                        ego_trans = carla.Transform(carla.Location(x=m_data['x'], y=m_data['y'], z=m_data['z']), carla.Rotation(yaw=m_data['yaw'], pitch=m_data['pitch'], roll=m_data['roll']))
                        parking_goal = [p_data['x'], p_data['y'], p_data['yaw']]
                        parking_goal = convert_slot_coord(ego_trans, parking_goal)
                        target_point_seq.append(parking_goal)
                self.target_point_seq.append(target_point_seq)

                self.ego_trans.append([ego_trans.location.x, ego_trans.location.y, ego_trans.rotation.yaw])
                self.task_idx.append(task_id)    

            if len(pose_episode) != 0:
                rewards, rtg = compute_shaped_rtg_with_terminal_bonus(pose_episode, x_thresh=1.0, y_thresh=0.6, theta_thresh=10.0, step_goal_bonus=0.05,
                                                                        terminal_bonus=3.0, w_x=0.003, w_y=0.003, w_theta=0.0005)
                
                rtg_windowed = make_rtg_windowed_array(rtg)
                self.acc_return.append(rtg_windowed)


        #
        # plt.figure(figsize=(10, 8))
        #
        # # 绘制plot_x的直方图
        # plt.subplot(3, 1, 1)  # 3行1列的第1个
        # plt.hist(self.plot_x, bins=20, alpha=0.7, label='X distribution')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of X')
        # plt.legend()
        #
        # # 绘制plot_y的直方图
        # plt.subplot(3, 1, 2)  # 3行1列的第2个
        # plt.hist(self.plot_y, bins=20, alpha=0.7, label='Y distribution')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Y')
        # plt.legend()
        #
        # # 绘制plot_yaw的直方图
        # plt.subplot(3, 1, 3)  # 3行1列的第3个
        # plt.hist(self.plot_yaw, bins=20, alpha=0.7, label='Yaw distribution')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.title('Histogram of Yaw')
        # plt.legend()
        #
        # # 调整子图间距
        # plt.tight_layout()
        #
        # # 显示图形
        # plt.show()

        self.task_idx = np.array(self.task_idx).astype(np.int)
        # print("Task ID: ", self.task_idx)

        self.front = np.array(self.front).astype(np.string_)
        self.front_left = np.array(self.front_left).astype(np.string_)
        self.front_right = np.array(self.front_right).astype(np.string_)
        self.back = np.array(self.back).astype(np.string_)
        self.back_left = np.array(self.back_left).astype(np.string_)
        self.back_right = np.array(self.back_right).astype(np.string_)

        self.front_depth = np.array(self.front_depth).astype(np.string_)
        self.front_left_depth = np.array(self.front_left_depth).astype(np.string_)
        self.front_right_depth = np.array(self.front_right_depth).astype(np.string_)
        self.back_depth = np.array(self.back_depth).astype(np.string_)
        self.back_left_depth = np.array(self.back_left_depth).astype(np.string_)
        self.back_right_depth = np.array(self.back_right_depth).astype(np.string_)

        self.topdown = np.array(self.topdown).astype(np.string_)

        self.speed = np.array(self.speed).astype(np.float32)
        self.acc_x = np.array(self.acc_x).astype(np.float32)
        self.acc_y = np.array(self.acc_y).astype(np.float32)

        self.control = np.array(self.control).astype(np.int64)
        self.waypoint = np.array(self.waypoint).astype(np.int64)

        self.throttle_brake = np.array(self.throttle_brake).astype(np.float32)
        self.steer = np.array(self.steer).astype(np.float32)
        self.reverse = np.array(self.reverse).astype(np.int64)

        self.delta_x_values = np.array(self.delta_x_values).astype(np.float32)
        self.delta_y_values = np.array(self.delta_y_values).astype(np.float32)
        self.delta_yaw_values = np.array(self.delta_yaw_values).astype(np.float32)

        self.target_point_pre = np.array(self.target_point).astype(np.float32)
        # if self.cfg.normalize_traj:
        #     self.target_point_pre = self.normalize_trajectories(self.target_point_pre)
        self.target_point = self.target_point_pre.copy()
        self.ego_trans = np.array(self.ego_trans).astype(np.float32)
        self.ego_motion_seq = np.array(self.ego_motion_seq).astype(np.float32)
        self.target_point_seq = np.array(self.target_point_seq).astype(np.float32)
        self.acc_return = np.vstack(self.acc_return).astype(np.float32)

        # traj_mean = np.mean(self.target_point, axis=0, keepdims=True)    # shape (1, 3)
        # traj_std = np.std(self.target_point, axis=0, keepdims=True) + 1e-8  # shape (1, 3), add epsilon to avoid /0
        # self.target_point = (self.target_point - traj_mean) / traj_std


        logger.info('Preloaded {} sequences', str(len(self.front)))
        # logger.info('Mean and std of trajectory points are %s, %s.' % (traj_mean, traj_std))

    def normalize_trajectories(self, traj):
        traj = traj / np.array([[10.0, 10.0, 180.0]]).astype(np.float32)
        return traj

    def relabel_goals(self, epoch):
        # Custom relabeling logic — example: small perturbation
        self.target_point = self.disturb_target_points(self.target_point, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0), yaw_range=(-45.0, 45.0))
        print(f"[CarlaDataset] Relabeled parking goals for epoch {epoch}")

    def keep_goals(self, epoch):
        self.target_point = self.target_point_pre

    def disturb_target_points(self, target_point, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), yaw_range=(-10.0, 10.0)):
        """
        Adds uniform noise to (x, y, yaw_deg).
        
        Parameters:
            target_point: numpy array of shape (N, 3)
            x_range, y_range: noise range in meters
            yaw_range: noise range in degrees
        
        Returns:
            disturbed_points: numpy array of same shape
        """
        noise_x = np.random.uniform(*x_range, size=(target_point.shape[0], 1)).astype(np.float32)
        noise_y = np.random.uniform(*y_range, size=(target_point.shape[0], 1)).astype(np.float32)
        noise_yaw = np.random.uniform(*yaw_range, size=(target_point.shape[0], 1)).astype(np.float32)

        noise = np.concatenate([noise_x, noise_y, noise_yaw], axis=1)
        disturbed = target_point + noise
        return disturbed

    def __len__(self):
        return len(self.front)

    def __getitem__(self, index):
        data = {}
        keys = ['image', 'depth', 'extrinsics', 'intrinsics', 'target_point', 'ego_motion', 'segmentation',
                'gt_control', 'gt_acc', 'gt_steer', 'gt_reverse','gt_waypoint','delta_x', 'delta_y', 'delta_yaw',
                'gt_target_point_traj', 'ego_trans_traj', 'gt_control_traj', 'gt_acc_traj', 'gt_steer_traj', 'gt_reverse_traj']
        for key in keys:
            data[key] = []

        # image & extrinsics & intrinsics
        images = [self.image_process(self.front[index])[0], self.image_process(self.front_left[index])[0],self.image_process(self.front_right[index])[0],
                  self.image_process(self.back[index])[0], self.image_process(self.back_left[index])[0],self.image_process(self.back_right[index])[0],]
        images = torch.cat(images, dim=0)
        data['image'] = images

        data['extrinsics'] = self.extrinsic
        data['intrinsics'] = self.intrinsic

        # depth
        depths = [get_depth(self.front_depth[index], self.image_crop),
                  get_depth(self.front_left_depth[index], self.image_crop),
                  get_depth(self.front_right_depth[index], self.image_crop),
                  get_depth(self.back_depth[index], self.image_crop),
                  get_depth(self.back_left_depth[index], self.image_crop),
                  get_depth(self.back_right_depth[index], self.image_crop),]
        depths = torch.cat(depths, dim=0)
        data['depth'] = depths

        # segmentation
        segmentation = self.semantic_process(self.topdown[index], scale=0.5, crop=200,
                                             target_slot=self.target_point[index])
        data['segmentation'] = torch.from_numpy(segmentation).long().unsqueeze(0)

        # target_point
        data['target_point'] = torch.from_numpy(self.target_point[index])

        # ego_motion
        speed = self.speed[index]
        ego_motion = np.column_stack((speed, self.acc_x[index], self.acc_y[index]))
        data['ego_motion'] = torch.from_numpy(ego_motion)
        data['ego_motion_seq'] = torch.from_numpy(self.ego_motion_seq[index])
        data['target_point_seq'] = torch.from_numpy(self.target_point_seq[index]) 

        # gt control token
        data['gt_control'] = torch.from_numpy(self.control[index])

        # gt control raw
        data['gt_acc'] = torch.from_numpy(self.throttle_brake[index])
        data['gt_steer'] = torch.from_numpy(self.steer[index])
        data['gt_reverse'] = torch.from_numpy(self.reverse[index])

        # gt waypoint token
        data['gt_waypoint'] = torch.from_numpy(self.waypoint[index])

        # gt waypoint raw
        data['delta_x'] = torch.from_numpy(self.delta_x_values[index])
        data['delta_y'] = torch.from_numpy(self.delta_y_values[index])
        data['delta_yaw'] = torch.from_numpy(self.delta_yaw_values[index])

        # accumulated reward
        data['acc_rew'] = torch.from_numpy(self.acc_return[index])

        # ego_trans
        data['ego_trans'] = torch.from_numpy(self.ego_trans[index])

        # Start diffusion Part
        start_task_idx, curr_task_idx = self.task_idx[index], self.task_idx[index]
        num_in_seq = 0

        while curr_task_idx == start_task_idx:
            data['gt_target_point_traj'].append(torch.from_numpy(self.target_point[index+num_in_seq]))
            data['ego_trans_traj'].append(torch.from_numpy(self.ego_trans[index+num_in_seq]))
            data['gt_control_traj'].append(torch.from_numpy(self.control[index+num_in_seq]))
            data['gt_acc_traj'].append(torch.from_numpy(self.throttle_brake[index+num_in_seq]))
            data['gt_steer_traj'].append(torch.from_numpy(self.steer[index+num_in_seq]))
            data['gt_reverse_traj'].append(torch.from_numpy(self.reverse[index+num_in_seq]))
            num_in_seq = num_in_seq + 1
            if index + num_in_seq < len(self):
                curr_task_idx = self.task_idx[index+num_in_seq]
            else:
                break

        data['gt_target_point_traj'], data['ego_trans_traj'], data['gt_control_traj'], data['gt_acc_traj'], data['gt_steer_traj'], data['gt_reverse_traj'] = torch.vstack(data['gt_target_point_traj']), torch.vstack(data['ego_trans_traj']), torch.vstack(data['gt_control_traj']), torch.vstack(data['gt_acc_traj']), torch.vstack(data['gt_steer_traj']), torch.vstack(data['gt_reverse_traj'])


        return data


    # def __getitem__(self, index):
    #     data = {}
    #     keys = ['image', 'depth', 'extrinsics', 'intrinsics', 'target_point', 'ego_motion', 'segmentation',
    #             'gt_control', 'gt_acc', 'gt_steer', 'gt_reverse','gt_waypoint','delta_x', 'delta_y', 'delta_yaw',]
    #     for key in keys:
    #         data[key] = []

    #     # image & extrinsics & intrinsics
    #     images = [self.image_process(self.front[index])[0], self.image_process(self.front_left[index])[0],self.image_process(self.front_right[index])[0],
    #               self.image_process(self.back[index])[0], self.image_process(self.back_left[index])[0],self.image_process(self.back_right[index])[0],]
    #     images = torch.cat(images, dim=0)
    #     data['image'] = images

    #     data['extrinsics'] = self.extrinsic
    #     data['intrinsics'] = self.intrinsic

    #     # depth
    #     depths = [get_depth(self.front_depth[index], self.image_crop),
    #               get_depth(self.front_left_depth[index], self.image_crop),
    #               get_depth(self.front_right_depth[index], self.image_crop),
    #               get_depth(self.back_depth[index], self.image_crop),
    #               get_depth(self.back_left_depth[index], self.image_crop),
    #               get_depth(self.back_right_depth[index], self.image_crop),]
    #     depths = torch.cat(depths, dim=0)
    #     data['depth'] = depths

    #     # segmentation
    #     segmentation = self.semantic_process(self.topdown[index], scale=0.5, crop=200,
    #                                          target_slot=self.target_point[index])
    #     data['segmentation'] = torch.from_numpy(segmentation).long().unsqueeze(0)

    #     # target_point
    #     data['target_point'] = torch.from_numpy(self.target_point[index])

    #     # ego_motion
    #     speed = self.speed[index]
    #     ego_motion = np.column_stack((speed, self.acc_x[index], self.acc_y[index]))
    #     data['ego_motion'] = torch.from_numpy(ego_motion)
    #     data['ego_motion_seq'] = torch.from_numpy(self.ego_motion_seq[index])
    #     data['target_point_seq'] = torch.from_numpy(self.target_point_seq[index]) 

    #     # gt control token
    #     data['gt_control'] = torch.from_numpy(self.control[index])

    #     # gt control raw
    #     data['gt_acc'] = torch.from_numpy(self.throttle_brake[index])
    #     data['gt_steer'] = torch.from_numpy(self.steer[index])
    #     data['gt_reverse'] = torch.from_numpy(self.reverse[index])

    #     # gt waypoint token
    #     data['gt_waypoint'] = torch.from_numpy(self.waypoint[index])

    #     # gt waypoint raw
    #     data['delta_x'] = torch.from_numpy(self.delta_x_values[index])
    #     data['delta_y'] = torch.from_numpy(self.delta_y_values[index])
    #     data['delta_yaw'] = torch.from_numpy(self.delta_yaw_values[index])

    #     # accumulated reward
    #     data['acc_rew'] = torch.from_numpy(self.acc_return[index])


    #     return data

class ProcessSemantic:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, image, scale, crop, target_slot):
        """
        Process original BEV ground truth image; return cropped image with target slot
        :param image: PIL Image or path to image
        :param scale: scale factor
        :param crop: image crop size
        :param target_slot: center location of the target parking slot in meters; vehicle frame
        :return: processed BEV semantic ground truth
        """

        # read image from disk
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert('L')

        # crop image
        cropped_image = scale_and_crop_image(image, scale, crop)

        # draw target slot on BEV semantic
        cropped_image = self.draw_target_slot(cropped_image, target_slot)

        # create a new BEV semantic GT
        h, w = cropped_image.shape
        vehicle_index = cropped_image == 75
        target_index = cropped_image == 255
        semantics = np.zeros((h, w))
        semantics[vehicle_index] = 1
        semantics[target_index] = 2
        # LSS method vehicle toward positive x-axis on image
        semantics = semantics[::-1]

        return semantics.copy()

    def draw_target_slot(self, image, target_slot):

        size = image.shape[0]

        # convert target slot position into pixels
        x_pixel = target_slot[0] / self.cfg.bev_x_bound[2]
        y_pixel = target_slot[1] / self.cfg.bev_y_bound[2]
        target_point = np.array([size / 2 - x_pixel, size / 2 + y_pixel], dtype=int)

        # draw the whole parking slot
        slot_points = []
        for x in range(-27, 28):
            for y in range(-15, 16):
                slot_points.append(np.array([x, y, 1, 1], dtype=int))

        # rotate parking slots points

        slot_trans = np.array(
            carla.Transform(carla.Location(), carla.Rotation(yaw=float(-target_slot[2]))).get_matrix())
        slot_points = np.vstack(slot_points).T
        slot_points_ego = (slot_trans @ slot_points)[0:2].astype(int)

        # get parking slot points on pixel frame
        slot_points_ego[0] += target_point[0]
        slot_points_ego[1] += target_point[1]

        # image[tuple(slot_points_ego)] = 255
        H, W = image.shape
        x, y = slot_points_ego
        valid_mask = (x>=0) & (x<H) & (y>=0) & (y<W)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        image[x_valid, y_valid] = 255

        return image


class ProcessImage:
    def __init__(self, crop):
        self.crop = crop

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

    def __call__(self, image):
        if isinstance(image, carla.Image):
            image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
            image = image[:, :, :3]
            image = image[:, :, ::-1]
            image = Image.fromarray(image)
        else:
            image = Image.open(image).convert('RGB')

        crop_image = scale_and_crop_image(image, scale=1.0, crop=self.crop)

        return self.normalise_image(np.array(crop_image)).unsqueeze(0), crop_image
