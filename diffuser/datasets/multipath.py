import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F
import copy
import random
# from collections import namedtuple
# from diffuser.utils.rendering import PrisonerRendererGlobe, PrisonerRenderer
# from diffuser.datasets.prisoner import pad_collate_detections, pad_collate_detections_repeat

global_device_name = "cpu"
global_device = torch.device("cpu")

class BallWheelchairJointDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 folder_path, 
                 horizon,
                 use_padding,
                 max_path_length,
                 dataset_type = "pixel",
                 include_start_detection = False,
                 condition_path = True,
                 max_detection_num = 32,
                 max_trajectory_length = 4320,
                 num_detections = 16,
                 train_mode="dynamic",
                 condition_mode = "pre",
                 prediction_mode = "2d",
                 ):

        print("Loading dataset from: ", folder_path) # tell me where you are loading the dataset

        self.condition_path = condition_path # do we use constraints in the training/testing
        print("Condition Path: ", self.condition_path)
        self.mode = condition_mode # condition on previous trajectory or post predicted trajectory?
        self.train_mode = train_mode # get wheelchair trajectory, ball trajectory or both?
        self.pred_mode = prediction_mode # do predictions in the image space or 3D space?

        
        self.dataset_type = dataset_type # set the dataset type - no use here
        self.use_padding = use_padding # do we want to pad the trajectory if the future points are insufficient
        self.observation_dim = 2 # any use here?
        self.horizon = horizon # how long we predict into the future?
        self.max_detection_num = max_detection_num # maximum size of the previous trajectory
        self.max_trajectory_length = max_trajectory_length # maximum total trajectory length in the dataset we can have, raise error when exceeded
        self.num_detections = num_detections # of no use here

        self.agent_locs = []
        self.process_first_file = True

        # INFO: load the timesteps and the padded ball-wheelchair poses into lists
        self._load_data(folder_path)

        self.max_path_length = max_path_length # maximum path length we are using in the prediction process
        self.include_start_detection = include_start_detection # include the start location as a condition
        self.indices = self.make_indices(self.path_lengths, horizon)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        # INFO: divide trajectory i into indices ((i, start_ind=min_start+0, start + horizon), (i, start_ind=min_start+1, start + horizon), ... (i, max_start, max_start + horizon))
        for i, path_length in enumerate(path_lengths):
            min_start = 0 # 0, self.max_detection_num
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(min_start, max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def _load_data(self, folder_path):

        # INFO: load each trajectory with into np_files list
        np_files = []
        fps = get_lowest_root_folders(folder_path)
        for fp in fps:
            for file_name in sorted(os.listdir(fp)):
                np_file = np.load(os.path.join(fp, file_name), allow_pickle=True)
                # print(np_file)
                # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
                np_files.append(np_file)
        self.set_normalization_factors()

        # INFO: load trajectories into lists
        for np_file in np_files:
            self._load_file(np_file)

        # INFO: print out max-min path lengths for debug
        print("Path Lengths: ")
        print(max(self.path_lengths), min(self.path_lengths))

        # INFO: print out the number of paths in the dataset
        self.path_num = len(np_files)
        print("Path Num: ")
        print(self.path_num)

        # INFO: load the detections into self.detected_dics and now we assume fully observable so every ball-wheelchair pose is in
        self.process_detections()

        # after processing detections, we can pad
        if self.use_padding:
            for i in range(len(self.agent_locs)):
                # INFO: pad at the end of agent_locs such that we can still draw self.horizon out even from the last step in the traj
                self.agent_locs[i] = np.pad(self.agent_locs[i], ((0, self.horizon), (0, 0)), 'edge')

    def set_normalization_factors(self):

        # INFO: set the size of the image space
        self.min_x = 0
        self.max_x = 1280
        self.min_y = 0
        self.max_y = 720

        # INFO: set the size of the court 3D space
        self.court_min_x = -5
        self.court_max_x = 5 # court length constraint
        self.court_min_y = -5
        self.court_max_y = 5 # court width constraint
        self.theta_min = 0
        self.theta_max = np.pi

    def normalize_2d(self, arr, last_dim):
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = arr[..., evens]
        arr[..., evens] = ((x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1

        y = arr[..., odds]
        arr[..., odds] = ((y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        return arr 

    def normalize_3d(self, arr):
        x_3d = arr[..., 0]
        arr[..., 0] = ((x_3d - self.court_min_x) / (self.court_max_x - self.court_min_x)) * 2 - 1

        y_3d = arr[..., 1]
        arr[..., 1] = ((y_3d - self.court_min_y) / (self.court_max_y - self.court_min_y)) * 2 - 1

        theta = arr[..., 2]
        arr[..., 2] = ((theta - self.theta_min) / (self.theta_max - self.theta_min)) * 2 - 1
        return arr 

    def unnormalize_2d(self, arr, last_dim):
        evens = np.arange(0, last_dim, 2)
        odds = np.arange(1, last_dim, 2)

        x = arr[..., evens]
        arr[..., evens] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = arr[..., odds]
        arr[..., odds] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y
        return arr

    def unnormalize_3d(self, arr):
        x_3d = arr[..., 0]
        arr[..., 0] = ((x_3d + 1) / 2) * (self.court_max_x - self.court_min_x) + self.court_min_x

        y_3d = arr[..., 1]
        arr[..., 1] = ((y_3d + 1) / 2) * (self.court_max_y - self.court_min_y) + self.court_min_y

        theta = arr[..., 2]
        arr[..., 2] = ((theta + 1) / 2) * (self.theta_max - self.theta_min) + self.theta_min

        return arr
    
    def normalize(self, arr):
        arr = copy.deepcopy(arr)
        last_dim = arr.shape[-1]
        if self.pred_mode == "2d":
            arr = self.normalize_2d(arr, last_dim)
        elif self.pred_mode == "3d":
            if last_dim == 2:
                arr = self.normalize_2d(arr, last_dim)             
            elif last_dim == 3:
                arr = self.normalize_3d(arr)
            elif last_dim == 7:
                part_2d = arr[..., :4]
                part_3d = arr[..., 4:]
                arr_2d = self.normalize_2d(part_2d, 4)
                arr_3d = self.normalize_3d(part_3d)
                arr = np.concatenate((arr_2d, arr_3d), axis=-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return arr

    def unnormalize(self, obs):
        obs = copy.deepcopy(obs)
        last_dim = obs.shape[-1]
        if self.pred_mode == "2d":
            obs = self.unnormalize_2d(obs, last_dim)

        elif self.pred_mode == "3d":
            print('last_dim',last_dim)
            if last_dim == 2:
                obs = self.unnormalize_2d(obs, last_dim)
            elif last_dim == 3:
                obs = self.unnormalize_3d(obs)
            elif last_dim == 5:
                part_2d = obs[..., :2]
                part_3d = obs[..., 2:]
                arr_2d = self.normalize_2d(part_2d, 2)
                arr_3d = self.normalize_3d(part_3d)
                obs = np.concatenate((arr_2d, arr_3d), axis=-1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return obs
    
    def unnormalize_single_dim(self, obs):
        x = obs[..., 0]
        obs[..., 0] = ((x + 1) / 2) * (self.max_x - self.min_x) + self.min_x

        y = obs[..., 1]
        obs[..., 1] = ((y + 1) / 2) * (self.max_y - self.min_y) + self.min_y

        return obs

    def process_detections(self):
        self.detected_dics = []
        # INFO: self.detected_locations is from dataset, it is indexed by (traj_id, detected_loc_id_in_each_traj)
        for detected_locs in self.detected_locations:
            indices = []
            detects = []
            for i in range(len(detected_locs)):
                loc = detected_locs[i]
                if 1: # we assume the ball-wheelchair trajectory is fully observable on the image
                    # INFO: we already normalized detections 
                    # loc[0:2] = loc[0:2] * 2 - 1
                    # loc[2:4] = loc[2:4] * 2 - 1
                    indices.append(i)
                    detects.append(loc)
            # INFO: stack all the detects and indices together and form the detected_dics. self.detected_dics structure dim is [traj_num, detect_num] with the structure
            # INFO: (cont'd) [[(indices_in_traj, detects), (indices_in_traj, detects), ..., (indices_in_traj, detects)], [], ..., []]
            detects = np.stack(detects, axis=0)
            indices = np.stack(indices, axis=0)
            self.detected_dics.append((indices, detects))
    
    def select_random_rows(self, array, n):
        b, m = array.shape

        if n >= b:
            return array

        indices = np.arange(b)
        np.random.shuffle(indices)

        selected_indices = indices[:n]
        remaining_indices = indices[n:]

        selected_rows = np.full((n, m), -np.inf)
        selected_rows[:len(selected_indices)] = array[selected_indices]

        result = np.copy(array)
        result[remaining_indices] = -np.inf

        return result

    def _load_file(self, file):

        timesteps = file["timestep_observations"]
        detected_locations = file["detected_locations"]
        ball_locs = np.float32(file["ball_locations"])
        if self.pred_mode == "2d":
            chair_locs = np.float32(file["chair_2d_locations"])
        elif self.pred_mode == "3d":
            chair_locs = np.float32(file["chair_3d_locations"])
        else:
            raise NotImplementedError
        agent_locs = [ball_locs, chair_locs]

        if chair_locs.shape[-1] == 4:
            print(file)
        
        # INFO: path length should be smaller than the limit
        path_length = len(agent_locs[0])
        if path_length > self.max_trajectory_length:
            raise ValueError("Path length is greater than max trajectory length")

        # INFO: normalize ball and wheelchair locations
        agents = []
        for i in range(len(agent_locs)):
            agent = self.normalize(agent_locs[i])
            agents.append(agent)
        locs_normalized = np.concatenate(agents, axis=1)
        detected_locations_normalized = self.normalize(detected_locations)

        # INFO: each trajectory is an element in self.agent_locs
        if self.process_first_file:
            self.process_first_file = False
            self.timesteps = timesteps
            self.agent_locs = [locs_normalized]
            self.detected_locations = [detected_locations_normalized]
            self.path_lengths = [path_length]
        else:
            self.agent_locs.append(locs_normalized)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.detected_locations.append(detected_locations_normalized)
            self.path_lengths.append(path_length)

    def convert_global_for_lstm(self, global_cond_idx, global_cond, start, mode="pre"):
        """ Convert the indices back to timesteps and concatenate them together"""
        if mode == "pre":
            # INFO: we predict the future traj based on at most previous self.max_detection_num steps
            detection_num = min(self.max_detection_num, len(global_cond_idx))
            global_cond_idx = global_cond_idx[-detection_num:]
            global_cond = global_cond[-detection_num:]
            if len(global_cond_idx) == 0:
                return -1 * torch.ones((1, 213)) 
            # INFO: calculate normalized steps before the start of the sampled traj
            global_cond_idx_adjusted = (start - global_cond_idx) / self.max_trajectory_length
            # INFO: concat global_cond_idx_adjusted with detects before the sampled traj
            global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        elif mode == "post":
            # INFO: we also constrain the max detection num when we are using the post ball estimation 
            detection_num = min(self.max_detection_num, len(global_cond_idx))
            global_cond_idx = global_cond_idx[:detection_num]
            global_cond = global_cond[:detection_num]
            # INFO: calculate normalized steps after the start of the sampled traj
            global_cond_idx_adjusted = - (start - global_cond_idx) / self.max_trajectory_length
            # INFO: concat global_cond_idx_adjusted with detects before the sampled traj
            global_cond = np.concatenate((global_cond_idx_adjusted[:, None], global_cond), axis=1)
        else:
            raise NotImplementedError
        return torch.tensor(global_cond).float()

    def get_conditions(self, path_ind, start, end, trajectories):
        '''
            condition on current observation for planning
        '''
        # INFO: self.detected_dics structure dim is [traj_num, detect_num] with the structure
        # INFO: (cont'd) [[([indices_in_traj], [detects]), ([indices_in_traj], [detects]), ..., ([indices_in_traj], [detects])], [], ..., []]
        detected_dic = self.detected_dics[path_ind]

        if self.mode == "pre":
            # subtract off the start and don't take anything past the end
            start_idx_find = np.where(detected_dic[0] >= start)[0]
            end_idx_find = np.where(detected_dic[0] < end)[0]
            # These are global conditions where the global_cond_idx is the 
            # integer index within the trajectory of where the detection occured

            # INFO: Take the detections before the start of the trajectory
            before_start_detects = np.where(detected_dic[0] <= start)[0]
            if len(before_start_detects) == 0:
                global_cond_idx = np.array([])
                global_cond = np.array([])
            else:
                # INFO: indices_in_traj before the sampled traj
                global_cond_idx = detected_dic[0][:before_start_detects[-1]+1]
                # INFO: detects before the sampled traj
                global_cond = detected_dic[1][:before_start_detects[-1]+1]

            # INFO: concatenation of time and detects
            detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, start, mode=self.mode)
        elif self.mode == "post":
            # INFO: Take the detections before the start of the trajectory
            after_start_detects = np.where(detected_dic[0] >= start)[0]            
            if len(after_start_detects) == 0:
                global_cond_idx = np.array([])
                global_cond = np.array([])
            else:
                # INFO: indices_in_traj before the sampled traj
                global_cond_idx = detected_dic[0][after_start_detects[0]:]
                # INFO: detects before the sampled traj
                global_cond = detected_dic[1][after_start_detects[0]:]
            # INFO: concatenation of time and detects
            detection_lstm = self.convert_global_for_lstm(global_cond_idx, global_cond, start, mode=self.mode)
        else:
            raise NotImplementedError



        if self.condition_path:
            # always include the start of the path
            if self.include_start_detection:
                if self.train_mode == "dynamic":
                    idxs = np.array([[0], [-1]])
                    detects = np.array([[trajectories[0]], [trajectories[-1]]])
                elif self.train_mode == "ball" or self.train_mode == "chair" or self.train_mode == "ball_chair":
                    idxs = np.array([[0]])
                    detects = np.array([[trajectories[0]]])                    
            else:
                raise NotImplementedError
        else:
            idxs = np.array([])
            detects = np.array([])

        return detection_lstm, (idxs, detects)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # INFO: each element in self.indices corresponds to one sample described by path_ind, start_ind_in_path, end_ind_in_path
        path_ind, start, end = self.indices[idx]

        # INFO: draw the sample described by path_ind, start_ind_in_path, end_ind_in_path OUT!
        trajectories = self.agent_locs[path_ind][start:end]

        # INFO: all_detections is the concatenation of time and detects
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)

        ball_chair_at_start = np.concatenate((np.array([0]), np.array(trajectories[0])))

        batch = (trajectories, all_detections, conditions, ball_chair_at_start)
        return batch

    def draw_with_path_point_ind(self, path_ind, start, start_loc):
        end = start + self.horizon

        # INFO: draw the sample described by path_ind, start_ind_in_path, end_ind_in_path OUT!
        trajectories = self.agent_locs[path_ind][start:end]

        # INFO: all_detections is the concatenation of time and detects; conditions is ???
        all_detections, conditions = self.get_conditions(path_ind, start, end, trajectories)
        ball_chair_at_start = np.concatenate((np.array([0]), np.array(trajectories[0])))
        if start_loc is not None:
            conditions[1][0,0,2:] = start_loc[0,-1,:].detach().cpu().numpy()
            ball_chair_at_start[-2:] = start_loc[0,-1,:].detach().cpu().numpy()
        

        batch = (trajectories, all_detections, conditions, ball_chair_at_start)
        return [batch]

    def ball_chair_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))

        if self.pred_mode == "2d":
            all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
        elif self.pred_mode == "3d":
            all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
        else:
            raise NotImplementedError 
        
        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_chair_collate_fn_repeat(self, batch):

        num_samples = 1

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples

        if self.pred_mode == "2d":
            all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
        elif self.pred_mode == "3d":
            all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
        else:
            raise NotImplementedError 
        
        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,:2] for i in range(len(data))]
        all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        conditions = [((conditions[i][0]), conditions[i][1][...,:2]) for i in range(len(conditions))]
        ball_chair_at_start = [ball_chair_at_start[i][:3] for i in range(len(ball_chair_at_start))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))
        # global_cond = torch.tensor(np.stack(global_cond, axis=0))

        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def ball_collate_fn_repeat(self, batch):
        num_samples = 20

        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only ball states 
        data = [data[i][:,:2] for i in range(len(data))]
        all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        conditions = [((conditions[i][0]), conditions[i][1][...,:2]) for i in range(len(conditions))]
        ball_chair_at_start = [ball_chair_at_start[i][:3] for i in range(len(ball_chair_at_start))]

        data = torch.tensor(np.stack(data, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples

        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        global_dict = {"detections": detections.to(global_device_name)}

        return data, global_dict, conditions

    def chair_collate_fn(self, batch):

        # INFO: trajectories, all_detections, conditions, ball_chair_at_start
        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError   
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]
        
        # INFO: stack the trajectory samples together
        data = torch.tensor(np.stack(data, axis=0))

        # INFO: pad with the largest time_len?
        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        if self.mode == "post":
            global_dict = {"detections": detections.to(global_device_name), "motions_start": torch.Tensor(ball_chair_at_start).to(global_device_name)}
        elif self.mode == "pre":
            global_dict = {"detections": detections.to(global_device_name)}
        else:
            raise NotImplementedError

        return data, global_dict, conditions

    def chair_collate_fn_repeat(self, batch):
        num_samples = 5

        (data, all_detections, conditions, ball_chair_at_start) = zip(*batch)

        # INFO: extract only wheelchair states 
        data = [data[i][:,2:] for i in range(len(data))]
        if self.mode == "post":
            all_detections = [all_detections[i][:,:3] for i in range(len(all_detections))]
        elif self.mode == "pre":
            if self.pred_mode == "2d":
                all_detections = [all_detections[i][:,:5] for i in range(len(all_detections))]
            elif self.pred_mode == "3d":
                all_detections = [torch.cat((all_detections[i][:,:3], all_detections[i][:,5:]), dim=-1) for i in range(len(all_detections))]
            else:
                raise NotImplementedError 
        else:
            raise NotImplementedError  
        conditions = [((conditions[i][0]), conditions[i][1][...,2:]) for i in range(len(conditions))]

        data = torch.tensor(np.stack(data, axis=0))

        data = data.repeat((num_samples, 1, 1))
        all_detections = list(all_detections) * num_samples
        conditions = list(conditions) * num_samples
        ball_chair_at_start = ball_chair_at_start * num_samples

        x_lens = [len(x) for x in all_detections]
        xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
        detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

        # Pass this to condition our models rather than pass them separately
        if self.mode == "post":
            ball_chair_at_start_tensor = torch.Tensor(ball_chair_at_start).to(global_device_name) if not isinstance(ball_chair_at_start[0], torch.Tensor) else torch.stack(ball_chair_at_start).float().to(global_device_name)
            global_dict = {"detections": detections.to(global_device_name), "motions_start": ball_chair_at_start_tensor}
        elif self.mode == "pre":
            global_dict = {"detections": detections.to(global_device_name)}
        else:
            raise NotImplementedError
        return data, global_dict, conditions

    def collate_fn_repeat(self):
        return pad_collate_detections_repeat

def pad_collate_detections(batch):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), "detections": detections.to(global_device_name), "motions_start": torch.Tensor(prisoner_at_start).to(global_device_name)}

    return data, global_dict, conditions

def pad_collate_detections_repeat(batch, num_samples):
    (data, global_cond, all_detections, conditions, prisoner_at_start) = zip(*batch)

    data = torch.tensor(np.stack(data, axis=0))
    global_cond = torch.tensor(np.stack(global_cond, axis=0))

    data = data.repeat((num_samples, 1, 1))
    global_cond = global_cond.repeat((num_samples, 1))
    all_detections = list(all_detections) * num_samples
    conditions = list(conditions) * num_samples

    x_lens = [len(x) for x in all_detections]
    xx_pad = pad_sequence(all_detections, batch_first=True, padding_value=0)
    detections = pack_padded_sequence(xx_pad, x_lens, batch_first=True, enforce_sorted=False).to(torch.float32)

    # Pass this to condition our models rather than pass them separately
    global_dict = {"hideouts": global_cond.to(global_device_name), 
        "motions_start": torch.Tensor(prisoner_at_start).to(global_device_name).repeat_interleave(repeats=num_samples, dim=0)}

    return data, global_dict, conditions

def get_lowest_root_folders(root_folder):
    lowest_folders = []
    
    # Get all items in the root folder
    items = os.listdir(root_folder)
    
    # Check if each item is a directory
    for item in items:
        item_path = os.path.join(root_folder, item)
        
        if os.path.isdir(item_path):
            # Recursively call the function for subfolders
            subfolders = get_lowest_root_folders(item_path)
            
            if not subfolders:
                # If there are no subfolders, add the current folder to the lowest_folders list
                lowest_folders.append(item_path)         
            lowest_folders.extend(subfolders)
    if len(lowest_folders) == 0:
        return [root_folder]
    return lowest_folders

def convert_wheelchair_2d_to_3d(image_points, projection_matrix):
    """
    Project 2D image points to 3D world coordinates (z=0) using a given projection matrix.

    Args:
    - image_points (np.array): Nx2 array containing 2D image coordinates.
    - projection_matrix (np.array): 3x4 projection matrix combining camera intrinsics and extrinsics.

    Returns:
    - world_points (np.array): Nx3 array containing 3D world coordinates.
    """

    # Get homography matrix from projection matrix (for z=0)
    homography = projection_matrix[:, [0, 1, 3]]  # Take columns 0, 1, and 3 (ignoring z)

    # Calculate the inverse homography matrix
    homography_inverse = np.linalg.inv(homography)

    # Convert 2D image points to homogeneous coordinates (add an extra dimension)
    num_points = image_points.shape[0]
    
    homogeneous_image_points = np.hstack((image_points, np.ones((num_points, 1))))
    # print("homogeneous_image_points = ", homogeneous_image_points)

    # Compute the 3D world points in homogeneous coordinates
    world_homogeneous = homography_inverse @ homogeneous_image_points.T  # 3xN

    # Normalize by the last row to convert to 3D world coordinates
    world_points = (world_homogeneous / world_homogeneous[2, :]).T  # Nx3

    # Set Z = 0 (currently it is normalized so it is set to 1)
    world_points = world_points[:, :2]

    return world_points


if __name__ == "__main__":
    print("Hello World.")