from gym.envs.registration import register

REGISTERED_ROBOT_ENVS = False


def register_robot_envs(n_substeps=20, observation_type='state+image', reward_type='dense', image_size=84, use_xyz=False):
	global REGISTERED_ROBOT_ENVS
	if REGISTERED_ROBOT_ENVS:	
		return

	register(
		id='-Grasp-v0',
		entry_point='env.robot.Grasp:GraspEnv',
		kwargs=dict(
			xml_path='robot/Grasp.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)


	REGISTERED_ROBOT_ENVS = True
