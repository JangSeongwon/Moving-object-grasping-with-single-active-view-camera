import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
import math
import mujoco_py
import os
import xml.etree.ElementTree as et
import gym
from gym import error, spaces
from gym.utils import seeding
import copy
import env.robot.gym_utils as utils # Modified some gym utils to incorporate multiple bodies in mocap
import cv2

import time
import matplotlib.pyplot as plt
import torch

DEFAULT_SIZE = 500

def get_full_asset_path(relative_path):
    return os.path.join(os.path.dirname(__file__), 'assets', relative_path)

class BaseEnv(robot_env.RobotEnv):
    """Superclass for all robot environments.
    """
    def __init__(

        self, model_path, cameras, n_substeps=20, gripper_rotation=[0,1,0,0], 
        has_object=False, image_size=84, reset_free=False, distance_threshold=0.01, action_penalty=0,
        observation_type='state+image', reward_type='dense', reward_bonus=True, use_xyz=False, action_scale=0.05, render=False
    ):
        """Initializes a new robot environment.
        Args:
            model_path (string): path to the environments XML file
            cameras (int): Camera/s to be used. Description in src/arguments.py
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_rotation (array): fixed rotation of the end effector, expressed as a quaternion
            has_object (boolean): whether or not the environment has an object
            image_size (int): size of image observations, if applicable
            reset_free (boolean): whether the arm configuration is reset after each episode
            distance_threshold (float): the threshold after which a goal is considered achieved
            action_penalty (float): scalar multiplier that penalizes high magnitude actions
            observation_type ('image' or 'state+image'): the observation type
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            reward_bonus (boolean): whether bonuses should be given for subgoals (only for dense rewards)
            use_xyz (boolean): whether movement is in 3d (xyz) or 2d (xy)
			action_scale (float): coefficient that scales scale position change
        """
        self.xml_dir = '/'.join(model_path.split('/')[:-1])
        self.reference_xml = et.parse(model_path)
        self.root = self.reference_xml.getroot()
        self.n_substeps = n_substeps
        self.gripper_rotation = np.array(gripper_rotation, dtype=np.float32)
        self.has_object = has_object
        self.distance_threshold = distance_threshold
        self.action_penalty = action_penalty
        self.observation_type = observation_type
        self.reward_type = reward_type
        self.image_size = image_size
        self.reset_free = reset_free
        self.reward_bonus = reward_bonus
        self.use_xyz = use_xyz
        self.action_scale = action_scale
        self.closed_angle = 0

        # Robot workspace configurations
        self.center_of_table = np.array([1.655, 0.3, 0.53625])
        self.default_z_offset = 0.04
        self.max_z = 1.0
        self.min_z = 0.6
        self.state_dim = 4 if use_xyz else 3

        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.state_space_shape = self.state_space.shape

        self.render_for_human = render
        self.cameras = cameras

        super(BaseEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos={})
    
    def goal_distance(self, goal1, goal2, use_xyz):
        assert goal1.shape == goal2.shape
        goal1 = np.around(goal1, 4)
        goal2 = np.around(goal2, 4)
        return np.linalg.norm(goal1 - goal2, axis=-1)

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        raise NotImplementedError('Reward signal has not been implemented for this task!')

    # Gripper helper
    # ----------------------------
    def _gripper_sync(self):
        self.sim.data.qpos[10] = 0.1

        self.sim.data.qpos[12] = -0.5

        # print(self.sim.data.qpos)
        # print('shape', self.sim.data.qpos.shape)
        
    def _gripper_consistent(self, angle):
        return 0
    
    # RobotEnv methods
    # ----------------------------
    def _step_callback(self):
        self.sim.forward()

    def _limit_gripper(self, gripper_pos, pos_ctrl):
        return pos_ctrl


    def _set_action(self, action):
        assert action.shape == (4,)

        action = action.copy() # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        # print('pos', pos_ctrl)
        # print('gripper', gripper_ctrl)
        self._pos_ctrl_magnitude = np.linalg.norm(pos_ctrl)

        # make sure gripper does not leave workspace
        gripper_pos = self.sim.data.get_site_xpos('grasp')
        pos_ctrl = self._limit_gripper(gripper_pos, pos_ctrl)

        pos_ctrl *= self.action_scale # limit maximum change in position
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])

        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, self.gripper_rotation, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


    def end_effector_pos(self):
        raise NotImplementedError('_get_achieved_goal has not been implemented for this task!')

    def _get_robot_state_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        eef_velp = self.sim.data.get_site_xvelp('grasp') * dt

        x = time.process_time()
        #print(x)
        sin = math.sin(x)
        self.sim.data.qvel[13:16] = np.array([ 0, -0.2, 0]) # -0.2, 0.14*sin
        self.sim.data.qvel[-3:] = np.random.randn(3) * 0
        # print('qvel', self.sim.data.qvel)
        # print('qvel_shape', self.sim.data.qvel.shape)

        eef_pos = self.sim.data.get_site_xpos('grasp')
        gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

        return np.concatenate([
            eef_pos, np.array([gripper_angle]) #eef_velp later add
        ], axis=0).astype(np.float32)

    def _get_image_obs(self):
        return self.render_obs(mode='rgb_array', width=self.image_size, height=self.image_size)

    def _get_obs(self):
        end_effector_position = self.end_effector_pos()
        obs = self._get_image_obs()
        state_ = self._get_robot_state_obs()

        return {
            'observation': obs,
            'achieved_goal': end_effector_position,
            'desired_goal': self.goal,
            'state': state_
        }


    def _render_callback(self):
        self.sim.forward()

    def _reset_sim(self):
        # Reset intial gripper position
        if not self.reset_free:
            self.sim.set_state(self.initial_state)
            self._sample_initial_pos()

        if self.has_object:
            self._sample_object_pos()

        self.sim.forward()
        return True

    def _sample_object_pos(self):
        raise NotImplementedError('_sample_object_pos has not been implemented for this task!')

    def _sample_goal(self, goal=None):
        assert goal is not None, 'must configure goal in task-specific class'
        self._pos_ctrl_magnitude = 0 # do not penalize at start of episode
        return goal

    def _sample_initial_pos(self, gripper_target=None):
        assert gripper_target is not None, 'must configure gripper in task-specific class'
        gripper_target[2] += 0.17 
        self.sim.data.set_mocap_pos('robot0:mocap2', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap2', self.gripper_rotation)
        self.sim.data.set_joint_qpos('right_outer_knuckle_joint', self.closed_angle)
        self._gripper_sync()
        for _ in range(10):
            self.sim.step()
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grasp').copy()
        self.init_finger_xpos = (self.sim.data.get_body_xpos('right_hand') + self.sim.data.get_body_xpos('left_hand'))/2

    def _is_success(self, achieved_goal, desired_goal):

        object_qpos = self.sim.data.get_joint_qpos('Object:joint')
        object_lift = object_qpos[2]
        self.grasp_lift = object_lift > 0.55

        if self.grasp_lift:
            d = self.goal_distance(achieved_goal, desired_goal, self.use_xyz)
            return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Sample initial position of gripper
        self._sample_initial_pos()
    
        # Extract information for sampling goals
        self.table_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('table')]

    def step(self, action):
        self.goal = self._sample_goal().copy()

        return super(BaseEnv, self).step(action)

    def render_video(self, mode=None, width=1080, height=1080, camera_id=None):
        data = []
        img = self.sim.render(width, height, camera_name='third_person', depth=False)[::-1, :, :]
        data.append(img)
        return np.asarray(data)

    def render(self, mode='human', width=500, height=500, depth=False, camera_id=0):
        return super(BaseEnv, self).render(mode, width, height)
