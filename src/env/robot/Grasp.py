import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path

from mujoco_py import MjViewer



class GraspEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='state+image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		self.sample_large = 1

		self.statefull_dim = (11,) if use_xyz else (8,)
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz,
		    has_object = True
		)

		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):

		reward = -30 # Per time period
		d = self.goal_distance(achieved_goal, goal, self.use_xyz)
		reaching_reward = 1 - np.tanh(4 * d)
		# print('reward', reaching_reward)
		reward += 10*reaching_reward

		"""gripper limit = 0.35"""
		gripper_distance1 = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')
		#gripper_distance2 = self.sim.data.get_joint_qpos('drive_joint')
		#dist = np.linalg.norm(gripper_distance1-gripper_distance2)
		#print(dist)
		# if gripper_distance1 > 0.32:
		# 		reward += 5
		if d < 0.1:
			ClosingGripperReward = np.tanh(8 * gripper_distance1)
			reward += 2*ClosingGripperReward

		ee_v = np.concatenate([self.sim.data.get_site_xvelp('grasp'), self.sim.data.get_site_xvelr('grasp')])
		vel = np.sum(abs(ee_v))/6
		vel_reward = 1 - np.tanh(20 * vel)
		#print('vel', vel)
		#print('vel_reward', vel_reward)
		if d < 0.3:
			reward += 2*vel_reward

		object_qpos = self.sim.data.get_joint_qpos('Object:joint')
		object_lift = object_qpos[2]
		#print(object_qpos)
		object_move = abs(object_qpos[0] - 1.6)
		#print(object_move)
		# if object_move > 0.05:
		# 	reward -= 3

		self.cube_geom_id = self.sim.model.geom_name2id("Object_target")
		self.l_finger_geom_ids = [self.sim.model.geom_name2id("j10")]
		self.r_finger_geom_ids = [self.sim.model.geom_name2id("j11")]
		Contact_Left_Gripper = False
		Contact_Right_Gripper = False

		for i in range(self.sim.data.ncon):
			N = self.sim.data.contact[i]
			if N.geom1 in self.l_finger_geom_ids and N.geom2 == self.cube_geom_id:
				Contact_Left_Gripper = True
			if N.geom1 == self.cube_geom_id and N.geom2 in self.l_finger_geom_ids:
				Contact_Left_Gripper = True
			if N.geom1 in self.r_finger_geom_ids and N.geom2 == self.cube_geom_id:
				Contact_Right_Gripper = True
			if N.geom1 == self.cube_geom_id and N.geom2 in self.r_finger_geom_ids:
				Contact_Right_Gripper = True
		self.has_grasp = Contact_Left_Gripper and Contact_Right_Gripper

		if self.has_grasp:
			reward += 5
			lift_reward = np.tanh(20*(object_lift-0.45))
			reward += 10*lift_reward
			if self._is_success(achieved_goal, goal):
				reward += 10

		return reward

	def _sample_object_pos(self):
		object_qpos = self.sim.data.get_joint_qpos('Object:joint')
		object_qvel = self.sim.data.get_joint_qvel('Object:joint')

	def end_effector_pos(self):
		return self.sim.data.get_site_xpos('grasp').copy()

	def _sample_goal(self):
		goal = self.sim.data.get_site_xpos('target')
		self.sim.forward()
		# print('goal', goal)
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = np.array([1.2561169, 0.3, 0.62603332])
		gripper_target[0] += self.np_random.uniform(-0.05, 0.1, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1, 0.1, size=1)
		gripper_target[2] += self.np_random.uniform(-0.05, 0.1, size=1)
		# print('gripper_target', gripper_target)
		BaseEnv._sample_initial_pos(self, gripper_target)
