import string
from typing import Optional, List

import gym
from gym.core import Env
import numpy as np
import os
import time
import copy
from components.objects import *

from tdw.output_data import OutputData, Images, CameraMatrices
from tdw.replicant.arm import Arm
from tdw.tdw_utils import TDWUtils
from components.avatar import TopDownAvatar, Avatar

from envs import CookController, GameController
from collections import Counter
import tdw, magnebot
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from PIL import Image

import json
import pickle
from functools import partial

from utils.utils import get_ego_topdown
from replicant import Replicant
import open3d as o3d
from copy import deepcopy

PLACE_INTO_PUZZLE = 1
PLACE_ON_THE_PLATE = 12
PLATE_DIST_THRESHOLD = 0.25

unconcerned_object_name = ["vk0007_steak_knife", "wood_board", "plate05",
						   "shapes_puzzle_physics_v1_piecemapped", "shapes_puzzle_physics_v2_piecemapped", "shapes_puzzle_physics_v3_piecemapped", "shapes_puzzle_physics_v4_piecemapped"]

def convert_pos(id, pos):
	if id == 0:
		return np.array([-pos[0], pos[1], -pos[2]])
	elif id == 1:
		return np.array([pos[2], pos[1], -pos[0]])
	elif id == 2:
		return pos
	elif id == 3:
		return np.array([-pos[2], pos[1], pos[0]])

class TDW(Env):
	def __init__(self, task, port=1071, number_of_agents=1, rank=0,
				 screen_size=1024, exp=False, launch_build=True, gt_mask=True,
				 enable_collision_detection=False, save_dir='results', max_steps=30,
				 data_prefix='dataset/', save_img=True, save_per_step=8,
				 is_test=False):
		self.key_ids = None
		self.cutting_board_pos = None
		self.last_joint_actions = None
		self.actions = None
		self.rejected_reason = None
		self.rejected = None
		self.task = task
		self.data_prefix = data_prefix
		self.replicant_colors = None
		self.replicant_ids = None
		self.names_mapping = None
		self.action_buffer = None
		self.object_manager = None
		self.gt_mask = gt_mask
		self.satisfied = None
		self.number_of_agents = number_of_agents
		self.seed = None
		self.rng = None
		self.num_step = 0
		self.reward = 0
		self.done = False
		self.exp = exp
		self.success = False
		self.num_frames = 0
		self.data_id = rank
		self.port = port
		self.screen_size = screen_size
		self.launch_build = launch_build
		self.enable_collision_detection = enable_collision_detection
		self.controller = None
		self.save_img = save_img
		self.save_per_step = save_per_step
		self.is_test = is_test
		object_space = gym.spaces.Dict({
			'id': gym.spaces.Discrete(30),
			# 'seg_color': gym.spaces.Box(0, 255, (3,), dtype=np.int32),
			'name': gym.spaces.Text(max_length=100, charset=string.printable),
			'pos': gym.spaces.Box(-30, 30, (3,), dtype=np.float32),  # to fix
			'upbound_pos': gym.spaces.Box(-30, 30, (3,), dtype=np.float32),
		})

		self.action_space_single = gym.spaces.Dict({
			'type': gym.spaces.Discrete(2),
			'obj_id': gym.spaces.Discrete(30),
			'obj_name': gym.spaces.Text(max_length=100, charset=string.printable),
			'pos': gym.spaces.Box(-30, 30, (3,), dtype=np.float32),
			'prompt': gym.spaces.Text(max_length=10000, charset=string.printable),
			'prompt_aug': gym.spaces.Text(max_length=10000, charset=string.printable),
		})

		self.observation_space_single = gym.spaces.Dict({
			'rgb': gym.spaces.Text(max_length=100, charset=string.printable),
			'depth':gym.spaces.Text(max_length=100, charset=string.printable),
			'ego_histories': gym.spaces.Tuple(gym.spaces.Text(max_length=100, charset=string.printable) for _ in range(self.save_per_step)),
			'objects': gym.spaces.Tuple(object_space for _ in range(30)),
			'last_joint_actions': gym.spaces.Text(max_length=1000, charset=string.printable),
			'rejected': gym.spaces.Discrete(2),
			'rejected_reason': gym.spaces.Text(max_length=1000, charset=string.printable),
			'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
			'camera_matrix': gym.spaces.Box(-30, 30, (self.save_per_step, 4, 4), dtype=np.float32),
		})

		self.observation_space = gym.spaces.Dict({
			str(i): self.observation_space_single for i in range(self.number_of_agents)
		})

		self.action_space = gym.spaces.Dict({
			str(i): self.action_space_single for i in range(self.number_of_agents)
		})

		self.max_steps = max_steps
		self.f = open(f'action{port}.log', 'w')
		self.action_list = []

		self.segmentation_colors = {}
		self.object_names = {}
		self.object_ids = {}
		self.held_objects = {i: None for i in range(self.number_of_agents)}
		self.fov = 0
		self.obs = None
		self.save_dir = save_dir

	def reset(
			self,
			*,
			seed: Optional[int] = None,
			options: Optional[dict] = None,
	):
		"""
        reset the environment
        options:
            output_dir: Optional[str] = None,
            save_img=True,
        """
		# Changes it to always, since in each step, we need to get the image
		if self.controller is not None:
			self.controller.communicate({"$type": "terminate"})
			self.controller.socket.close()
		if self.task == 'cook':
			assert self.number_of_agents == 2, "agent number not match!"
			self.controller = CookController(port=self.port, check_version=True, launch_build=self.launch_build,
											 screen_width=self.screen_size, screen_height=self.screen_size,
											 enable_collision_detection=self.enable_collision_detection,
											 logger_dir=options['output_dir'])
		elif self.task == 'game' or self.task == 'game_3' or self.task == 'game_2':
			if self.task == 'game':
				assert self.number_of_agents == 4, "agent number not match!"
			elif self.task == 'game_3':
				assert self.number_of_agents == 3, "agent number not match!"
			elif self.task == 'game_2':
				assert self.number_of_agents == 2, "agent number not match!"

			self.controller = GameController(port=self.port, check_version=True, launch_build=self.launch_build,
											 screen_width=self.screen_size, screen_height=self.screen_size,
											 enable_collision_detection=self.enable_collision_detection,
											 logger_dir=options['output_dir'], number_of_agents=self.number_of_agents)

			
		
		print("Controller connected")
		self.success = False
		# self.messages = [None for _ in range(self.number_of_agents)]
		self.reward = 0
		self.save_img = options['save_img']
		self.satisfied = {}
		if options['output_dir'] is not None: self.save_dir = options['output_dir']
		super().reset(seed=seed)
		self.seed = seed
		self.rng = np.random.RandomState(seed)

		info = self.controller.setup(seed=seed, number_of_agents=self.number_of_agents, is_test=self.is_test)

		print("setup completed!")

		if self.task == 'game' or self.task == 'game_3' or self.task == 'game_2':
			for agent in self.controller.agents:
				self.controller.communicate({"$type": "set_field_of_view",
											"avatar_id": str(agent.replicant_id), "field_of_view": 82})
			self.fov = 82

		else:
			for agent in self.controller.agents:
				self.controller.communicate({"$type": "set_field_of_view",
											"avatar_id": str(agent.replicant_id), "field_of_view": 75})
			self.fov = 75

		self.num_step = 0
		self.num_frames = 0

		self.done = False
		self.rejected = [False for _ in range(self.number_of_agents)]
		self.rejected_reason = [None for _ in range(self.number_of_agents)]
		self.action_buffer = [[] for _ in range(self.number_of_agents)]
		self.held_objects = {i: None for i in range(self.number_of_agents)}
		self.camera_matrix_dict = dict()
		self.flag = False

		for id, agent in enumerate(self.controller.agents):
			agent.look_at(target=None)
		
		while True:
			finished = True
			for id, agent in enumerate(self.controller.agents):
				ActionStatus = tdw.replicant.action_status.ActionStatus

				if agent.action.status == ActionStatus.ongoing:
					finished = False
					continue

			if finished:
				break

			data = self.controller.communicate([])

		if self.save_img:  # store the final frame a.k.a the observation for next step
			all_camera_matrices = dict()
			for i in range(len(data) - 1):
				r_id = OutputData.get_data_type_id(data[i])
				if r_id == 'imag':
					images = Images(data[i])
					avatar_id = images.get_avatar_id()
					TDWUtils.save_images(images=images, filename=f"{0:05d}",
										 output_directory=os.path.join(self.save_dir, avatar_id))
				elif r_id == 'cama':
					camera_matrices = CameraMatrices(data[i])
					camera_matrix = camera_matrices.get_camera_matrix()
					avatar_id = camera_matrices.get_avatar_id()
					# print(avatar_id, camera_matrix)
					all_camera_matrices[avatar_id] = camera_matrix

			self.camera_matrix_dict[0] = all_camera_matrices
			
		for i, agent in enumerate(self.controller.agents):
			id = str(i)
			ego_topdown = get_ego_topdown(self.task, self.camera_matrix_dict[0][id], os.path.join(self.save_dir, id, f"img_{self.num_frames:05d}.png"), os.path.join(self.save_dir, id, f"depth_{self.num_frames:05d}.png"))
			if 'train' in self.save_dir or 'test' in self.save_dir: # exclude combo from saving this image
				Image.fromarray(ego_topdown).save(os.path.join(self.save_dir, id, f"reconstructed_{self.num_frames:05d}.png"))

		self.key_ids = [self.num_frames] * 8
		self.obs = self.get_obs()
		info["held_objects"] = self.held_objects
		self.controller.get_reward(info) # set the first reward as steps needed
		for obj in self.obs["0"]['objects']:
			if obj['name'] == 'wood_board':
				self.cutting_board_pos = obj['pos']
				break
		# info = {
		# 	'recipe': self.controller.recipe if hasattr(self.controller,
		# 												'recipe') else self.controller.puzzle_id2piece_id,
		# 	'reachable_region': self.controller.reachable_region,
		# 	'reachable_bins': self.controller.reachable_bins,
		# }
			
		return self.obs, info

	def get_obs(self):
		obs = {str(i): {} for i in range(self.number_of_agents)}
		# print(self.last_camera_matrices['top_down'])
		for i, agent in enumerate(self.controller.agents):
			id = str(i)
			obs[id]['rgb'] = os.path.join(self.save_dir, id, f"img_{self.num_frames:05d}.png")
			obs[id]['depth'] = os.path.join(self.save_dir, id, f"depth_{self.num_frames:05d}.png")
			obs[id]['ego_histories'] = [os.path.join(self.save_dir, id, f"img_{i:05d}.png") for i in self.key_ids]
			obs[id]['camera_matrix'] = [self.camera_matrix_dict[frame_id][id] for frame_id in self.key_ids]
			# obs[id]['top_down'] = os.path.join(self.save_dir, id, f"top_down_{self.num_frames:05d}.png")
			obs[id]['objects'] = self.controller.get_objects()
			
			obs[id]['rejected'] = self.rejected[i]
			obs[id]['rejected_reason'] = self.rejected_reason[i]
			obs[id]['last_joint_actions'] = self.last_joint_actions
			# print(id, obs[id]['objects'])
			while len(obs[id]['objects']) < 30:
				obs[id]['objects'].append({
					'id': None,
					'name': None,
					'pos': None,
					"upbound_pos": None
				})

			obs[id]['FOV'] = self.fov
		return obs

	def get_with_character_mask(self, agent_id, character_object_ids):
		color_set = [self.segmentation_colors[id] for id in character_object_ids if id in self.segmentation_colors] + [
			self.replicant_colors[id] for id in character_object_ids if id in self.replicant_colors]
		curr_with_seg = np.zeros_like(self.obs[str(agent_id)]['seg_mask'])
		curr_seg_flag = np.zeros((self.screen_size, self.screen_size), dtype=bool)
		for i in range(len(color_set)):
			color_pos = (self.obs[str(agent_id)]['seg_mask'] == np.array(color_set[i])).all(axis=2)
			curr_seg_flag = np.logical_or(curr_seg_flag, color_pos)
			curr_with_seg[color_pos] = color_set[i]
		return curr_with_seg, curr_seg_flag

	def get_2d_distance(self, pos1, pos2):
		return np.linalg.norm(np.array(pos1) - np.array(pos2))

	def get_2d_distance_except_height(self, pos1, pos2):
		return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2)

	def check_goal(self):
		r'''
        Check if the goal is achieved
        return: count, total, done
        '''
		return self.controller.check_goal()

	def step(self, actions):
		'''
		Run one timestep of the environment's dynamics
		'''
		start = time.time()
		finish = False
		num_frames = 0
		# time.sleep(1000)
		step_images: List[dict] = []
		step_camera_matrices: List[dict] = []
		self.actions = deepcopy(actions)

		for id, agent in enumerate(self.controller.agents):
			if actions[str(id)]['type'] is None:
				actions[str(id)].update(self.controller.parse_text_action(actions[str(id)]["prompt"], id))

		for id, agent in enumerate(self.controller.agents):
			self.rejected[id] = False
			self.rejected_reason[id] = None
			action = actions[str(id)]

			if "reject" in action and action["reject"]:
				self.rejected[id] = True
				self.rejected_reason[id] = f"The prompt format of agent {id} is wrong"
				continue

			if action["type"] == "place" and (self.held_objects[id] is None or self.held_objects[id] != action["obj_id"]):
				self.rejected[id] = True
				self.rejected_reason[id] = f"agent {id} places when not holding an object or holding a wrong object"

			if action["type"] == "pick" and self.held_objects[id] is not None:
				self.rejected[id] = True
				self.rejected_reason[id] = f"agent {id} picks when holding an object"
			
			if action["type"] == "cut" and self.held_objects[id] is not None:
				self.rejected[id] = True
				self.rejected_reason[id] = f"agent {id} cuts when holding an object"

			if action["type"] == "cut" and self.get_2d_distance_except_height(self.cutting_board_pos, action["pos"]) > 0.1:
				self.rejected[id] = True
				self.rejected_reason[id] = f"agent {id} cuts an object that is not on the cutting board!"
				
			if action["type"] == "pick" or action["type"] == "place":
				if "place_type" in action and action["place_type"] == PLACE_INTO_PUZZLE:
					pos = action['pos'] + convert_pos(id, self.controller.relative_position_dict[action["obj_id"]])
				else:
					pos = action['pos']
				if not self.within_reach(id, pos):
					self.rejected[id] = True
					self.rejected_reason[id] = f"{action['type']} out of reachable region"

			if action["type"] == "place" and action["place_type"] != PLACE_INTO_PUZZLE and action["place_type"] != PLACE_ON_THE_PLATE:
				objects = self.controller.get_objects()
				for obj in objects:
					if obj["name"] is None or obj["name"] in unconcerned_object_name:
						continue

					if self.get_2d_distance(obj["pos"], action["pos"]) < 0.1:
						self.rejected[id] = True
						self.rejected_reason[id] = f"{action['type']} collide with object {obj['name']}"
						break

			if action["type"] == "pick": # check whether the object is in the puzzle box or plate
				pos = action['pos']
				flag = True
				for bin in self.controller.reachable_bins[id]:
					if self.get_2d_distance_except_height(pos, bin) < 0.1:
						flag = False
						break
				
				if self.task == "cook" and self.get_2d_distance_except_height(pos, self.cutting_board_pos) < 0.1:
					flag = False

				if flag:
					self.rejected[id] = True
					self.rejected_reason[id] = f"pick an object on the plate or on the puzzle box"

			if action["type"] == "place":
				if action["place_type"] == PLACE_INTO_PUZZLE and action["obj_id"] not in self.controller.puzzle_id2piece_id[self.controller.puzzles[id].id]:
					self.rejected[id] = True
					self.rejected_reason[id] = f"place a piece into the puzzle box that is not belong to the puzzle box"

				if action["place_type"] == PLACE_ON_THE_PLATE:
					num = [0, 0]
					ln = [0, 0]
					(num[0], num[1]), (ln[0], ln[1]), done = self.check_goal()
					if num[id] >= ln[id] or self.controller.recipe[id][num[id]] != action["obj_name"]:
						self.rejected[id] = True
						self.rejected_reason[id] = f"place an object that is not belong to the plate or in the wrong order"


		critical_position = [None for _ in range(self.number_of_agents)]
		for id, agent in enumerate(self.controller.agents):
			action = actions[str(id)]
			if "critical_position" in action:
				critical_position[id] = action["critical_position"]
			elif action["type"] == "pick" or action["type"] == "place" or action["type"] == "cut":
				if "place_type" in action and action["place_type"] == PLACE_INTO_PUZZLE:
					pos = action['pos'] + convert_pos(id, self.controller.relative_position_dict[action["obj_id"]])
				else:
					pos = action['pos']

				critical_position[id] = pos

		ids = [x for x in range(self.number_of_agents)]
		self.rng.shuffle(ids)
		current_critical_positions = []
		for id in ids:
			if self.rejected[id]:
				continue

			action = actions[str(id)]
			if action['type'] == 'wait':
				continue

			if critical_position[id] is not None:
				flag = False
				for pos in current_critical_positions:
					if self.get_2d_distance_except_height(pos, critical_position[id]) < 0.3:
						flag = True
						break
				
				if flag:
					self.rejected[id] = True
					self.rejected_reason[id] = f"collision with other agent"
				else:
					current_critical_positions.append(critical_position[id])

		for id, agent in enumerate(self.controller.agents):
			if self.rejected[id]:
				self.actions[str(id)]= {"type": "wait", "prompt": "wait"}
				continue

			action = actions[str(id)]

			if action['type'] == 'pick':
				self.held_objects[id] = action['obj_id']
				if "lift_up" in action:
					action["pos"][1] += 0.3
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'navigate_to'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'pick'})
					# self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'reset_arm'})
					self.action_buffer[id].append(
						{**copy.deepcopy(action), 'type': 'navigate_to', 'pos': agent.home_position})
				else:
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'navigate_to'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'pick'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'reset_arm'})
					self.action_buffer[id].append(
						{**copy.deepcopy(action), 'type': 'navigate_to', 'pos': agent.home_position})
					
				self.action_buffer[id].append({'type': 'reset_head'})	
				self.action_buffer[id].append({'type': 'look_at', 'pos': agent.home_look_at})

			elif action["type"] == 'place':  # drop held object in arm
				self.held_objects[id] = None
				if "lift_up" in action:
					action["pos"][1] += 0.3
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'navigate_to'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'place'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'reset_arm'})
					self.action_buffer[id].append(
						{**copy.deepcopy(action), 'type': 'navigate_to', 'pos': agent.home_position})
				else:
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'navigate_to'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'place'})
					self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'reset_arm'})
					self.action_buffer[id].append(
						{**copy.deepcopy(action), 'type': 'navigate_to', 'pos': agent.home_position})
				
				self.action_buffer[id].append({'type': 'reset_head'})
				self.action_buffer[id].append({'type': 'look_at', 'pos': agent.home_look_at})
			elif action["type"] == 'cut':
				self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'navigate_to'})
				self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'pick', 'obj_id': self.controller.knife_id, "set_kinematic_state": True, "pick_knife": True})
				self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'cut'})
				self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'set_manager'})
				self.action_buffer[id].append(
					{**copy.deepcopy(action), 'type': 'place', 'obj_id': self.controller.knife_id, 'pos': self.controller.knife_home_pos, "set_kinematic_state": True, "place_knife": True})
				self.action_buffer[id].append({**copy.deepcopy(action), 'type': 'reset_arm'})
				self.action_buffer[id].append(
						{**copy.deepcopy(action), 'type': 'navigate_to', 'pos': agent.home_position})
				
				self.action_buffer[id].append({'type': 'reset_head'})
				self.action_buffer[id].append({'type': 'look_at', 'pos': agent.home_look_at})
			elif action["type"] == 'wait':
				pass
			else:
				raise Exception("Unknown action type")


		while True:  # continue until all agent's action finishes
			finish = True
			for id, agent in enumerate(self.controller.agents):
				if isinstance(agent, Replicant):
					ActionStatus = tdw.replicant.action_status.ActionStatus
					arm = tdw.replicant.arm.Arm.left
				else:
					ActionStatus = magnebot.ActionStatus
					arm = magnebot.Arm.left

				if agent.action.status == ActionStatus.ongoing:  # ignore still dropping/unexpected collision
					finish = False
					continue
				assert agent.action.status == ActionStatus.success or agent.action.status == ActionStatus.still_dropping or ActionStatus.collision, f"Agent {id} action {agent.action} status is {agent.action.status}"
				if isinstance(agent, magnebot.Magnebot):
					print(agent.action.status)
				if len(self.action_buffer[id]) > 0:
					action = self.action_buffer[id].pop(0)
					finish = False
					if action['type'] == 'reach_for':
						agent.reach_for(action['pos'], arm)
					elif action['type'] == 'pick':
						if "pick_knife" in action:
							if id == 0:
								agent.pick_up(int(action["obj_id"]), arm=arm, reach_pos = np.array(self.controller.object_manager.transforms[action["obj_id"]].position) + np.array([0.1, 0, 0]),
					 					set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), object_rotate_angle = 90)
							else:
								agent.pick_up(int(action["obj_id"]), arm=arm, reach_pos = np.array(self.controller.object_manager.transforms[action["obj_id"]].position) + np.array([0.1, 0, 0]),
					 					set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]))
						elif isinstance(agent, Replicant) and "offset" in action:
							if "lift_up" in action:
								agent.pick_up(int(action["obj_id"]), arm=arm, offset=action["offset"],
					  						reach_pos = np.array(self.controller.object_manager.transforms[action["obj_id"]].position) + np.array([0, 0.3, 0]),
					 						set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]))
							else:
								agent.pick_up(int(action["obj_id"]), arm=arm, offset=action["offset"],
					 						set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]))
						elif "lift_up" in action:
							agent.pick_up(int(action["obj_id"]), arm=arm,
					 					reach_pos = np.array(self.controller.object_manager.transforms[action["obj_id"]].position) + np.array([0, 0.3, 0]),
					 					set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]))
						else:
							agent.pick_up(int(action["obj_id"]), arm=arm,
					 						set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]))
					elif action['type'] == 'place':
						if "place_knife" in action:
							if id == 0:
								agent.place(action['pos'], arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True, object_rotate_angle = 270)
							else:
								agent.place(action['pos'], arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True)
								
						elif "place_type" in action and action["place_type"] == PLACE_INTO_PUZZLE:
							if self.task == "cook":
								agent.place(action['pos'] + convert_pos(id, self.controller.relative_position_dict[action["obj_id"]]),
										arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True)
							else:
								agent.place(action['pos'] + convert_pos(id, self.controller.relative_position_dict[action["obj_id"]]),
										arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True)
						else:
							if self.task == "cook":
								agent.place(action['pos'], arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True)
							else:
								agent.place(action['pos'], arm, action["obj_id"], set_kinematic_state=("set_kinematic_state" in action and action["set_kinematic_state"]), skip_drop = True)

					elif action['type'] == 'reset_arm':
						agent.reset_arm(arm)
					elif action['type'] == 'navigate_to':
						if 'pos' in action:
							if "tag" in action:
								# print(agent.replicant_id, [agent.home_position[0], action['pos'][1], action["pos"][2]])
								agent.navigate_to(
									np.array([agent.home_position[0], action['pos'][1], action["pos"][2]]))
							else:
								agent.navigate_to(
									np.array([action['pos'][0], action['pos'][1], agent.home_position[2]]))
						else:
							agent.navigate_to(action['obj_id'])
					elif action['type'] == "cut":
						sliced = None
						if action["obj_name"] == "whole_cheese":
							sliced = CheeseSlice(position=TDWUtils.array_to_vector3(action["pos"]), scale={"x": 2.0, "y": 2.0, "z": 2.0})
						elif action["obj_name"] == "whole_tomato":
							sliced = TomatoSlice(position=TDWUtils.array_to_vector3(action["pos"]), scale={"x": 3, "y": 3, "z": 3})
						elif action["obj_name"] == "whole_onion":
							sliced = OnionSlice(position=TDWUtils.array_to_vector3(action["pos"]), scale={"x": 2.5, "y": 2.5, "z": 2.5})
						else:
							raise NotImplementedError
						
						agent.slice(target=action["obj_id"], num_times=3, sliced_target=sliced, object_manager = self.controller.object_manager)
						
					elif action['type'] == 'set_manager':
						self.controller.objects = [obj for obj in self.controller.objects if obj.id != action["obj_id"]]
						self.controller.objects.append(sliced)
						for agent_d in self.controller.agents:
							agent_d.collision_detection.exclude_objects.append(sliced.id)

					elif action['type'] == 'reset_head':
						agent.reset_head()

					elif action['type'] == 'look_at':
						agent.look_at(target=None)

					elif action['type'] == 'wait':
						pass
					else:
						raise Exception("Unknown action type")

			# for id, agent in enumerate(self.controller.agents):
			# 	print(id, agent.action.status)
			if finish:
				break

			num_frames += 1
			data = self.controller.communicate([])

			cur_images = {}
			cur_camera_matrices = {}
			for i in range(len(data) - 1):
				r_id = OutputData.get_data_type_id(data[i])
				if r_id == 'imag':
					images = Images(data[i])
					avatar_id = images.get_avatar_id()
					cur_images[avatar_id] = images
					# TDWUtils.save_images(images=images, filename=f"{self.num_frames + num_frames:05d}",
					# 						output_directory=os.path.join(self.save_dir, avatar_id))\
				elif r_id == "cama":
					camera_matrices = CameraMatrices(data[i])
					camera_matrix = camera_matrices.get_camera_matrix()
					avatar_id = camera_matrices.get_avatar_id()
					# print(avatar_id, camera_matrix)

					cur_camera_matrices[avatar_id] = camera_matrix

			step_images.append(cur_images)
			step_camera_matrices.append(cur_camera_matrices)
			
			
		add_frames = []
		if len(step_images) > 0:
			if self.save_img:  # store the final frame a.k.a the observation for next step
				ids = np.round(np.linspace(0, 1, self.save_per_step) * (len(step_images) - 1)).astype(np.int32).tolist()
				self.key_ids = [self.num_frames]
				# ids = [x for x in range(len(step_images))]
				top_down = {}
				for avatar, image in step_images[0].items():
					if avatar == 'top_down':
						continue
					if 'train' in self.save_dir or 'test' in self.save_dir:
						top_down[avatar] = np.array(Image.open(f"{self.save_dir}/{avatar}/reconstructed_{self.num_frames:05d}.png"))

				for i in ids[1:]: # skip the first one, since it's the same with the last frame of last step
					if i >= len(step_images):
						continue
					self.key_ids.append(self.num_frames + i + 1)
					add_frames.append(self.num_frames + i + 1)
					self.camera_matrix_dict[self.num_frames + i + 1] = step_camera_matrices[i]
					for avatar, image in step_images[i].items():
						TDWUtils.save_images(images=image, filename=f"{self.num_frames + i + 1:05d}",
											 output_directory=os.path.join(self.save_dir, avatar))
						if 'train' in self.save_dir or 'test' in self.save_dir:  # exclude combo from saving this image
							if avatar == 'top_down':
								continue
							cur_top_down = get_ego_topdown(self.task, step_camera_matrices[i][avatar], f"{self.save_dir}/{avatar}/img_{self.num_frames + i + 1:05d}.png", f"{self.save_dir}/{avatar}/depth_{self.num_frames + i + 1:05d}.png")
							Image.fromarray(cur_top_down).save(
								os.path.join(self.save_dir, avatar, f"reconstructed_{self.num_frames + i + 1:05d}.png"))
							top_down[avatar] = top_down[avatar] * (cur_top_down == 0) + cur_top_down
				
				for avatar, image in top_down.items():
					Image.fromarray(image).save(os.path.join(self.save_dir, avatar, f"overlay_{self.num_frames + num_frames:05d}.png"))
		

		self.num_frames += num_frames
		self.action_list.append(self.actions)
		_, _, self.success = self.check_goal()
		for id, agent in enumerate(self.controller.agents):
			action = self.actions[str(id)]
			task_status = agent.action.status
			self.f.write('step: {}, action: {}, time: {}, status: {}\n'
						 .format(self.num_step, action["type"],
								 time.time() - start,
								 task_status))
			self.f.write('position: {}, forward: {}\n'.format(
				agent.dynamic.transform.position,
				agent.dynamic.transform.forward, ))
			self.f.flush()
			# if task_status != ActionStatus.success and task_status != ActionStatus.ongoing:
			# 	reward -= 0.5

		self.num_step += 1

		done = False
		if self.num_step >= self.max_steps or self.success:
			done = True
			self.done = True

		self.last_joint_actions = self.controller.convert_actions_prompt(self.actions)
		obs = self.get_obs()

		info = {'done': done,
				'num_frames_for_step': num_frames,
				'num_step': self.num_step,
				'held_objects': self.held_objects
				}
		reward, prompt_value = self.controller.get_reward(info)
		self.reward += reward
		info['prompt_value'] = prompt_value
		if self.save_img:
			info['camera_matrices'] = dict()

			if not self.flag:
				info['camera_matrices'][0] = self.camera_matrix_dict[0]
				self.flag = True

			for i in add_frames:
				info['camera_matrices'][i] = self.camera_matrix_dict[i]
			

		if done:
			info['reward'] = self.reward
			info['success'] = self.success

		self.obs = obs
		return self.obs, reward, done, info

	def convert_actions_prompt(self, actions):
		return self.controller.convert_actions_prompt(actions)


	def within_reach(self, id, pos):
		return self.controller.reachable_region[id][0] < pos[0] < self.controller.reachable_region[id][2] and self.controller.reachable_region[id][1] < pos[2] < \
			self.controller.reachable_region[id][3]

	def close(self):
		print('close')
		with open(f'action.pkl', 'wb') as f:
			d = {'seed': self.seed, 'actions': self.action_list}
			pickle.dump(d, f)
