import numpy as np
from copy import deepcopy
def l2_dist(pos1, pos2):
	if isinstance(pos1, dict):
		pos1 = np.array([pos1["x"], pos1["y"], pos1["z"]])
	if isinstance(pos2, dict):
		pos2 = np.array([pos2["x"], pos2["y"], pos2["z"]])

	return np.linalg.norm(pos1 - pos2)

from tdw.tdw_utils import TDWUtils

BIN_DIST_THRESHOLD = 0.1
BOARD_DIST_THRESHOLD = 0.2
PLATE_DIST_THRESHOLD = 0.25
WHOLE_TO_SLICE = {"whole_cheese": "cheese_slice", "whole_tomato": "tomato_slice", "whole_onion": "onion_slice"}
SLICE_TO_WHOLE = {v: k for k, v in WHOLE_TO_SLICE.items()}

def get_transformed_slice_name(name):
	if name in WHOLE_TO_SLICE.keys():
		return WHOLE_TO_SLICE[name]

	return name

def get_transformed_whole_name(name):
	if name in SLICE_TO_WHOLE.keys():
		return SLICE_TO_WHOLE[name]

	return name

PLACE_ON_THE_CUTTING_BOARD = 11
PLACE_ON_THE_PLATE = 12
PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT = 13
PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT = 14
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT = 15
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT = 16

class CookPlanAgent:
	def __init__(self, agent_id, logger, output_dir='results', is_altruism = False):
		self.object_on_cutting_board = None
		self.id2pos = None
		self.name2id = None
		self.agent_id = agent_id
		self.agent_type = 'test_agent'
		self.local_step = 0
		self.is_reset = False
		self.logger = logger
		self.map_id = 0
		self.output_dir = output_dir
		self.last_action = None
		self.obs = None
		self.save_img = True
		self.object_in_hand = None
		self.recipe = None
		self.reachable_region = None
		self.is_altruism = is_altruism

	def reset(self, obs, info, output_dir='results'):
		self.local_step = 0
		self.is_reset = True
		self.output_dir = output_dir
		self.last_action = None
		self.recipe = info["recipe"].copy()
		self.reachable_region = info["reachable_region"][int(self.agent_id)].copy()
		self.oppo_reachable_region = info["reachable_region"][1 - int(self.agent_id)].copy()
		self.private_region = info["private_region"][int(self.agent_id)].copy()
		self.oppo_private_region = info["private_region"][1 - int(self.agent_id)].copy()
		if int(self.agent_id) == 0:
			self.oppo_reachable_region[1] = -5.0
		else:
			self.oppo_reachable_region[2] = 5.0

		self.seed = info["seed"]
		self.rng = np.random.RandomState(self.seed + self.agent_id)
		self.reachable_bins = info["reachable_bins"][int(self.agent_id)].copy()
		self.object_in_hand = None
		self.last_object_in_hand = None
		self.is_oppo = False
		self.is_immediate = False
		self.place_pos = None
		self.last_place_pos = None
		self.place_type = None
		self.dest_plates = info["dest_plates"].copy()

	def pick_policy(self, self_needed, oppo_needed, is_immediate, obs, num, force_selfish = False):
		if not self.is_altruism or force_selfish:
			if self_needed is not None and self.in_private_region(self_needed["pos"]):
				# If the current object the agent needs is in the private region, then go and get it
				action = {"type": "pick", "obj_id": self_needed["id"], "obj_name": self_needed["name"], 
							"pos": self_needed["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
				self.object_in_hand = self_needed
				if self_needed["name"] in WHOLE_TO_SLICE.keys():
					self.place_pos = self.id2upbound[self.name2id["wood_board"]]
					self.place_type = PLACE_ON_THE_CUTTING_BOARD
				else:
					if num == 0:
						self.place_pos = self.dest_plates[int(self.agent_id)]
					else:
						for all_objs in obs["objects"]:
							if all_objs["name"] is not None and all_objs["name"] == self.recipe[int(self.agent_id)][num - 1]:
								self.place_pos = all_objs["upbound_pos"]
								break

					self.place_type = PLACE_ON_THE_PLATE
						
			elif oppo_needed is not None:
				# pick the object that opponent needs and ready to give it to the opponent
				action = {"type": "pick", "obj_id": oppo_needed["id"], "obj_name": oppo_needed["name"], 
							"pos": oppo_needed["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
				self.place_pos = self.id2upbound[self.name2id["wood_board"]]
				self.object_in_hand = oppo_needed
				self.is_oppo = True
				self.is_immediate = is_immediate
				self.place_type = PLACE_ON_THE_CUTTING_BOARD

			else:
				# If the current object the agent needs is not in the private region, then wait
				action = {"type": "wait"}

		else:
			if oppo_needed is not None and (self.get_opponent_objects(obs) < 4 or (is_immediate and self.get_opponent_objects(obs) < 5) or self_needed is None or not self.in_private_region(self_needed["pos"])):
				# pick the object that opponent needs and ready to give it to the opponent
				action = {"type": "pick", "obj_id": oppo_needed["id"], "obj_name": oppo_needed["name"], 
							"pos": oppo_needed["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
				self.place_pos = self.id2upbound[self.name2id["wood_board"]]
				self.object_in_hand = oppo_needed
				self.is_oppo = True
				self.is_immediate = is_immediate
				self.place_type = PLACE_ON_THE_CUTTING_BOARD

			elif self_needed is not None and self.in_private_region(self_needed["pos"]):
				# If the current object the agent needs is in the private region, then go and get it
				action = {"type": "pick", "obj_id": self_needed["id"], "obj_name": self_needed["name"], 
							"pos": self_needed["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
				self.object_in_hand = self_needed
				if self_needed["name"] in WHOLE_TO_SLICE.keys():
					self.place_pos = self.id2upbound[self.name2id["wood_board"]]
					self.place_type = PLACE_ON_THE_CUTTING_BOARD
				else:
					if num == 0:
						self.place_pos = self.dest_plates[int(self.agent_id)]
					else:
						for all_objs in obs["objects"]:
							if all_objs["name"] is not None and all_objs["name"] == self.recipe[int(self.agent_id)][num - 1]:
								self.place_pos = all_objs["upbound_pos"]
								break

					self.place_type = PLACE_ON_THE_PLATE

			else:
				# If the current object the agent needs is not in the private region, then wait
				action = {"type": "wait"}

		return action

	def get_opponent_objects(self, obs):
		board_pos = self.id2pos[self.name2id["wood_board"]]
		res = 0
		for obj in obs["objects"]:
			if obj["name"] is not None and self.within_oppo_reach(obj["pos"]) and \
				l2_dist(obj["pos"], self.dest_plates[1 - int(self.agent_id)]) > PLATE_DIST_THRESHOLD and \
				l2_dist(obj["pos"], board_pos) > BOARD_DIST_THRESHOLD:
				res += 1
		
		return res

	def act(self, obs):
		if obs["rejected"]:
			self.place_pos = deepcopy(self.last_place_pos)
			self.object_in_hand = deepcopy(self.last_object_in_hand)
			self.place_type = deepcopy(self.last_place_type)
			self.is_oppo = deepcopy(self.last_is_oppo)
			self.is_immediate = deepcopy(self.last_is_immediate)

		self.last_is_oppo = deepcopy(self.is_oppo)
		self.last_is_immediate = deepcopy(self.is_immediate)
		self.last_place_pos = deepcopy(self.place_pos)
		self.last_object_in_hand = deepcopy(self.object_in_hand)
		self.last_place_type = deepcopy(self.place_type)
		action = None
		self.id2name = {obj['id']: obj['name'] for obj in obs["objects"]}
		self.name2id = {obj['name']: obj['id'] for obj in obs["objects"]}
		self.id2pos = {obj['id']: obj['pos'] for obj in obs["objects"]}
		self.id2upbound = {obj['id']: obj['upbound_pos'] for obj in obs["objects"]}
		self.object_on_cutting_board = self.get_item_on_cutting_board(obs)
		num = [0, 0]
		ln = [0, 0]
		(num[0], num[1]), (ln[0], ln[1]), done = self.check_goal(obs)
		if num[int(self.agent_id)] == 0:
			self.plate_top = None
		else:
			self.plate_top = "the " + self.recipe[int(self.agent_id)][num[int(self.agent_id)] - 1]

		rest_recipe = []
		for i in range(num[int(self.agent_id)], ln[int(self.agent_id)]):
			rest_recipe.append(self.recipe[int(self.agent_id)][i])
		
		self_needed = None
		if len(rest_recipe) > 0:
			for obj in obs["objects"]:
				if (obj["name"] == get_transformed_whole_name(rest_recipe[0]) or obj["name"] == rest_recipe[0]) and self.within_reach(obj["pos"]):
					self_needed = obj
					break

					
		oppo_needed = None
		oppo_id = 1 - int(self.agent_id)
		is_immediate = False # Does the opponent need the object immediately?

		for i in range(num[oppo_id], ln[oppo_id]):
			for obj in obs["objects"]:
				if (obj["name"] == get_transformed_whole_name(self.recipe[oppo_id][i]) or obj["name"] == self.recipe[oppo_id][i]) and self.in_private_region(obj["pos"]):
					oppo_needed = obj
					if i == num[oppo_id]:
						is_immediate = True
					break

			if oppo_needed is not None:
				break

		self.obj_in_bins = [None for _ in range(len(self.reachable_bins))]
		for obj in obs["objects"]:
			if obj["id"] is None:
				continue
			for i in range(len(self.reachable_bins)):
				if l2_dist(obj["pos"], self.reachable_bins[i]) < BIN_DIST_THRESHOLD:
					self.obj_in_bins[i] = obj
					break

		# print(self.agent_id, end = " ")
		# for i in range(len(self.reachable_bins)):
		# 	print(obj_in_bins[i], end=" ")

		# print(end = '\n')
		possible_actions = [{"type": "wait", "prompt": "wait"}]
		if self.object_in_hand is None:
			for i in range(len(self.reachable_bins)):
				if self.obj_in_bins[i] is not None:
					action = {"type": "pick", "obj_id": self.obj_in_bins[i]["id"], "obj_name": self.obj_in_bins[i]["name"],
							  "pos": self.obj_in_bins[i]["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
					action["prompt"] = self.action2prompt(action)
					possible_actions.append(action)
		
			if self.object_on_cutting_board is not None:
				action = {"type": "pick", "obj_id": self.object_on_cutting_board["id"], "obj_name": self.object_on_cutting_board["name"],
							  "pos": self.object_on_cutting_board["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
				action["prompt"] = self.action2prompt(action)
				possible_actions.append(action)

				if self.object_on_cutting_board["name"] in WHOLE_TO_SLICE.keys():
					action = {"type": "cut", "obj_id": self.object_on_cutting_board["id"], "obj_name": self.object_on_cutting_board["name"],
							  "pos": self.object_on_cutting_board["pos"]}
					action["prompt"] = self.action2prompt(action)
					possible_actions.append(action)

		else:
			for i in range(len(self.reachable_bins)):
				if self.obj_in_bins[i] is None:
					pos = deepcopy(self.reachable_bins[i])
					action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"],
							  "pos": np.array(pos), "place_type": PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT + i, "set_kinematic_state": True}
					action["prompt"] = self.action2prompt(action)
					possible_actions.append(action)
		
			if self.object_on_cutting_board is None:
				pos = deepcopy(self.id2upbound[self.name2id["wood_board"]])
				action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"],
							  "pos": np.array(pos), "place_type": PLACE_ON_THE_CUTTING_BOARD, "set_kinematic_state": True}
				action["prompt"] = self.action2prompt(action)
				possible_actions.append(action)

			
			if num[int(self.agent_id)] < len(self.recipe[int(self.agent_id)]) and self.object_in_hand["name"] == self.recipe[int(self.agent_id)][num[int(self.agent_id)]]:
				place_plate_pos = self.dest_plates[int(self.agent_id)]
				action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"],
								"pos": place_plate_pos, "place_type": PLACE_ON_THE_PLATE, "set_kinematic_state": True}
				action["prompt"] = self.action2prompt(action)
				possible_actions.append(action)
				
		if self.object_on_cutting_board is not None:
			name = get_transformed_slice_name(self.object_on_cutting_board["name"])
			if name in rest_recipe:
				if self.object_in_hand is not None and self.object_in_hand["name"] != name:
					if self.place_type == PLACE_ON_THE_PLATE:
						# place the current object on hand to the plate
						action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"], 
									"pos": self.place_pos, "place_type": self.place_type, "set_kinematic_state": True}
					else:
						# place the current object in hand down to the private region
						put_id = None
						for i in range(len(self.reachable_bins)):
							if self.obj_in_bins[i] is None:
								put_id = i
								break

						if put_id is None:
							action = {"type": "wait", "prompt": "wait"}
						else:
							self.place_pos = self.reachable_bins[put_id]
							self.place_pos[0] += self.rng.uniform(-0.02, 0.02)
							self.place_pos[2] += self.rng.uniform(-0.02, 0.02)
							if put_id == 0:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT
							elif put_id == 1:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT
							elif put_id == 2:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT
							elif put_id == 3:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT
							else:
								raise AssertionError("error in placing object in private region!")
							
							action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"], 
										"pos": self.place_pos, "place_type": self.place_type, "set_kinematic_state": True}

					self.object_in_hand = None
					self.place_pos = None
					self.place_type = None
					self.is_oppo = False
					self.is_immediate = False
					

				elif self.object_on_cutting_board["name"] in WHOLE_TO_SLICE.keys():
					# need to slice this object
					action = {"type": "cut", "obj_id": self.object_on_cutting_board["id"], "obj_name": self.object_on_cutting_board["name"], 
			   					"pos": self.object_on_cutting_board["pos"]}
				
				else:
					# go and grasp this object
					if name == rest_recipe[0]:
						# need it now!, put it into the plate
						self.object_in_hand = self.object_on_cutting_board
						if num[int(self.agent_id)] == 0:
							self.place_pos = self.dest_plates[int(self.agent_id)]
						else:
							for all_objs in obs["objects"]:
								if all_objs["name"] is not None and all_objs["name"] == self.recipe[int(self.agent_id)][num[int(self.agent_id)] - 1]:
									self.place_pos = all_objs["upbound_pos"]
									break

						self.place_type = PLACE_ON_THE_PLATE
						
						action = {"type": "pick", "obj_id": self.object_on_cutting_board["id"], "obj_name": self.object_on_cutting_board["name"],
					  			"pos": self.object_on_cutting_board["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}

					else:
						# need it later, put it into the private region
						put_id = None
						for i in range(len(self.reachable_bins)):
							if self.obj_in_bins[i] is None:
								put_id = i
								break
						
						if put_id is None:
							# can't get it, must clear the private region
							action = self.pick_policy(self_needed, oppo_needed, is_immediate, obs, num[int(self.agent_id)], force_selfish = True)
							action["set_kinematic_state"] = True
						else:
							# put the object to the private region
							self.object_in_hand = self.object_on_cutting_board
							self.place_pos = self.reachable_bins[put_id]
							self.place_pos[0] += self.rng.uniform(-0.02, 0.02)
							self.place_pos[2] += self.rng.uniform(-0.02, 0.02)
							if put_id == 0:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT
							elif put_id == 1:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT
							elif put_id == 2:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT
							elif put_id == 3:
								self.place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT
							else:
								raise AssertionError("error in placing object in private region!")
							action = {"type": "pick", "obj_id": self.object_on_cutting_board["id"], "obj_name": self.object_on_cutting_board["name"],
					  			"pos": self.object_on_cutting_board["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}

			else:
				if self.object_in_hand is not None:
					# place the current object to the destination(if it is the cutting board, then wait)
					if l2_dist(self.place_pos, self.id2pos[self.name2id["wood_board"]]) < BOARD_DIST_THRESHOLD:
						action = {"type": "wait"}
					else:
						action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"], 
								"pos": self.place_pos, "place_type": self.place_type, "set_kinematic_state": True}
						
						self.object_in_hand = None
						self.place_pos = None
						self.place_type = None
						self.is_oppo = False
						self.is_immediate = False

				else:
					action = self.pick_policy(self_needed, oppo_needed, is_immediate, obs, num[int(self.agent_id)])

		else:
			if self.object_in_hand is not None:
				# place the current object to the destination
				if self.is_oppo:
					if (is_immediate or self.get_opponent_objects(obs) < 4) and self.get_opponent_objects(obs) < 5:
						action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"], 
								"pos": self.place_pos, "set_kinematic_state": True, "place_type": self.place_type}
						self.object_in_hand = None
						self.place_pos = None
						self.is_oppo = False
						self.is_immediate = False
						self.place_type = None
					else:
						action = {"type": "wait"}

				else:
					action = {"type": "place", "obj_id": self.object_in_hand["id"], "obj_name": self.object_in_hand["name"], 
							"pos": self.place_pos, "set_kinematic_state": True, "place_type": self.place_type}
					self.object_in_hand = None
					self.place_pos = None
					self.place_type = None
					self.is_oppo = False
					self.is_immediate = False
			else:
				action = self.pick_policy(self_needed, oppo_needed, is_immediate, obs, num[int(self.agent_id)])		

		if int(self.agent_id) == 1 and action["type"] != "wait":
			action["tag"] = True

		action["prompt"] = self.action2prompt(action)
		action["prompt_proposer"] = self.convert_prompt_proposer([possible_action["prompt"] for possible_action in possible_actions], action["prompt"])
		self.last_action = action

		return action
		
	def within_reach(self, pos):
		return self.reachable_region[0] < pos[0] < self.reachable_region[2] and self.reachable_region[1] < pos[2] < \
			self.reachable_region[3]
	
	def within_oppo_reach(self, pos):
		return self.oppo_reachable_region[0] < pos[0] < self.oppo_reachable_region[2] and self.oppo_reachable_region[1] < pos[2] < \
			self.oppo_reachable_region[3]
	
	def in_private_region(self, pos):
		return self.private_region[0] < pos[0] < self.private_region[2] and self.private_region[1] < pos[2] < \
			self.private_region[3]

	def get_item_on_cutting_board(self, obs):
		board_pos = self.id2pos[self.name2id["wood_board"]]

		for obj in obs["objects"]:
			if obj["name"] is not None and obj['name'] != "wood_board" and obj['name'] != "vk0007_steak_knife" and l2_dist(obj['pos'], board_pos) < BOARD_DIST_THRESHOLD:
				return obj
		
		return None

	def check_goal(self, obs):
		plate1_pos = self.dest_plates[0]
		num1 = len(self.recipe[0])
		last_height = -5
		for i in range(len(self.recipe[0])):
			name = self.recipe[0][i]
			objs = []
			for obj in obs["objects"]:
				if obj["name"] == name and l2_dist(obj["pos"], plate1_pos) < PLATE_DIST_THRESHOLD and obj["pos"][1] > last_height:
					objs.append(obj)

			if len(objs) == 0:
				num1 = i
				break

			if len(objs) > 1:
				objs = sorted(objs, key=lambda x: x["pos"][1])

			last_height = objs[0]["pos"][1]

		last_height = -5
		plate2_pos = self.dest_plates[1]
		num2 = len(self.recipe[1])
		for i in range(len(self.recipe[1])):
			name = self.recipe[1][i]
			objs = []
			for obj in obs["objects"]:
				if obj["name"] == name and l2_dist(obj["pos"], plate2_pos) < PLATE_DIST_THRESHOLD and obj["pos"][1] > last_height:
					objs.append(obj)

			if len(objs) == 0:
				num2 = i
				break

			if len(objs) > 1:
				objs = sorted(objs, key=lambda x: x["pos"][1])

			last_height = objs[0]["pos"][1]

		return (num1, num2), (len(self.recipe[0]), len(self.recipe[1])), (num1 == len(self.recipe[0]) and num2 == len(self.recipe[1]))
	
	def action2prompt(self, action):		
		if action["type"] == "pick":
			return f"pick up the {action['obj_name']}"
		elif action["type"] == "cut":
			return f"cut the {action['obj_name']}"
		elif action["type"] == "place":
			ret = f"place the {action['obj_name']} onto the "
			if action["place_type"] == PLACE_ON_THE_CUTTING_BOARD:
				ret += "cutting board"
			elif action["place_type"] == PLACE_ON_THE_PLATE:
				ret += "plate"
			elif action["place_type"] == PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT:
				ret += "top left corner of the private region"
			elif action["place_type"] == PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT:
				ret += "top right corner of the private region"
			elif action["place_type"] == PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT:
				ret += "bottom left corner of the private region"
			elif action["place_type"] == PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT:
				ret += "bottom right corner of the private region"
			else:
				raise NotImplementedError(f"Unknown place type {action['place_type']}")
			return ret
		elif action["type"] == "wait":
			return "wait"
		else:
			raise NotImplementedError(f"Unknown action type {action['type']}")
	
	def convert_prompt_proposer(self, possible_actions, chosen_action):
		assert chosen_action in possible_actions, f"Chosen action is not in possible actions"
		on_cutting_board_name = 'the ' + self.object_on_cutting_board["name"] if self.object_on_cutting_board is not None else 'nothing'
		reachable_object_names = []
		for obj in self.obj_in_bins:
			if obj is not None:
				reachable_object_names.append(obj["name"])
		if self.object_on_cutting_board is not None:
			reachable_object_names.append(self.object_on_cutting_board["name"])
		reachable_objects = ', '.join(reachable_object_names)
		if reachable_objects == '':
			reachable_objects = 'nothing'
		
		obj_in_hand = 'nothing'
		if self.last_object_in_hand is not None:
			obj_in_hand = self.last_object_in_hand["name"]
		
		return f"I'm holding {obj_in_hand}. I can see {reachable_objects} in my reachable region, {on_cutting_board_name} is on the cutting board, {self.plate_top if self.plate_top is not None else 'nothing'} is on the top of the plate. My possible actions are: {', '.join(possible_actions)}. I choose to {chosen_action}."