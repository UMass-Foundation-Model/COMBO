import numpy as np
import random
from copy import deepcopy
def l2_dist(pos1, pos2):
	return np.linalg.norm(np.array(pos1) - np.array(pos2))

BIN_THRESHOLD = 0.2
IN_PUZZLE_THRESHOLD = 0.1
PLACE_INTO_PUZZLE = 1
PLACE_ON_THE_LEFT_BORDER = 2
PLACE_ON_THE_RIGHT_BORDER = 3
PLACE_INSIDE_PRIVATE_AREA_LEFT = 4
PLACE_INSIDE_PRIVATE_AREA_RIGHT = 5

check_ord = [1, 2, 0, 3]

# caution: fix is_clockwise=True for nearest direction, otherwise the map_direction dict would be wrong
map_direction = [{0: "random", 1: 0, 2: "random", 3: 3},
				 {0: 3, 1: "random", 2: 0, 3: "random"},
				 {0: "random", 1: 3, 2: "random", 3: 0},
				 {0: 0, 1: "random", 2: 3, 3: "random"}]

def convert_pos(id, pos):
	if id == 0:
		return np.array([-pos[0], pos[1], -pos[2]])
	elif id == 1:
		return np.array([pos[2], pos[1], -pos[0]])
	elif id == 2:
		return pos
	elif id == 3:
		return np.array([-pos[2], pos[1], pos[0]])

class GamePlanAgent:
	def __init__(self, agent_id, logger, output_dir='results', admit_error_placing=False, fix_clockwise=None, random_dir=False, nearest_dir=False):
		self.id2name = None
		self.annotation_dict = None
		self.semantic_name_dict = None
		self.relative_position_dict = None
		self.place_type = None
		self.puzzle_ids = None
		self.reachable_bins = None
		self.id2pos = None
		self.name2id = None
		self.agent_id = agent_id
		self.agent_type = 'test_agent'
		self.logger = logger
		self.map_id = 0
		self.output_dir = output_dir
		self.last_action = None
		self.obs = None
		self.save_img = True
		self.object_in_hand = None
		self.puzzle_id2piece_id = None
		self.reachable_region = None
		self.pos_to_place = None
		self.admit_error_placing = admit_error_placing
		self.fix_clockwise = fix_clockwise
		self.random_dir = random_dir
		self.nearest_dir = nearest_dir

	def reset(self, obs, info, output_dir='results'):
		self.output_dir = output_dir
		self.last_action = None
		self.name2id = {obj['name']: obj['id'] for obj in obs["objects"]}
		self.puzzle_id2piece_id = info["puzzle_id2piece_id"].copy()
		self.puzzle_ids = list(self.puzzle_id2piece_id.keys())
		self.reachable_region = info["reachable_region"][int(self.agent_id)].copy()
		self.reachable_bins = info["reachable_bins"][int(self.agent_id)].copy()
		self.relative_position_dict = info["relative_position_dict"].copy()
		self.semantic_name_dict = info["semantic_name_dict"].copy()
		self.annotation_dict = info["annotation_dict"].copy()
		self.is_clockwise = info["is_clockwise"].copy()
		self.number_of_agents = len(info["reachable_bins"])

		if self.fix_clockwise is not None:
			self.is_clockwise = self.fix_clockwise

		if self.nearest_dir: 
			self.is_clockwise = True

		if self.is_clockwise == False:
			self.reachable_bins.reverse()

		self.seed = info["seed"]
		self.rng = np.random.RandomState(self.seed + self.agent_id)
		self.object_in_hand = None
		self.pos_to_place = None
		self.place_type = None
		self.wrong_piece_id = None
		self.place_wrong = False

	def pick_policy(self, can_reach_ids, num_bins, can_get_0, can_get_3, is_clockwise):
		if is_clockwise:
			if can_reach_ids[num_bins - 1] is None:
				# If the last bin is empty, then put a piece in it(pass them to the next agent)
				for i in range(num_bins - 1):
					if can_reach_ids[i] is not None and (i != 0 or can_get_0):
						action = {"type": "pick", "obj_id": can_reach_ids[i], "obj_name": self.id2name[can_reach_ids[i]], "pos": np.array(self.id2pos[can_reach_ids[i]]), "lift_up": True, "set_kinematic_state": True}
						self.object_in_hand = can_reach_ids[i]
						self.pos_to_place = self.reachable_bins[num_bins - 1]
						if self.is_clockwise:
							self.place_type = PLACE_ON_THE_LEFT_BORDER
						else:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER

						break
						
			if self.object_in_hand is None:
				# If bin 0 is not empty, then put it to an empty bin inside the reachable area(free the place for other agent to pass other pieces)
				if can_reach_ids[0] is not None and can_get_0:
					for i in range(1, num_bins - 1):
						if can_reach_ids[i] is None:
							action = {"type": "pick", "obj_id": can_reach_ids[0], "pos": np.array(self.id2pos[can_reach_ids[0]]), "lift_up": True, "set_kinematic_state": True}
							self.object_in_hand = can_reach_ids[0]
							self.pos_to_place = self.reachable_bins[i]
							if (i == 1 and self.is_clockwise) or (i == 2 and not self.is_clockwise):
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
							else:
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT

							break
							
			if self.object_in_hand is None:
				action = {"type": "wait"}

		else:
			if can_reach_ids[0] is None:
				# If the first bin is empty, then put a piece in it(pass them to the next agent)
				for i in range(num_bins - 1, 0, -1):
					if can_reach_ids[i] is not None and (i != 3 or can_get_3):
						action = {"type": "pick", "obj_id": can_reach_ids[i], "obj_name": self.id2name[can_reach_ids[i]], "pos": np.array(self.id2pos[can_reach_ids[i]]), "lift_up": True, "set_kinematic_state": True}
						self.object_in_hand = can_reach_ids[i]
						self.pos_to_place = self.reachable_bins[0]
						if self.is_clockwise:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER
						else:
							self.place_type = PLACE_ON_THE_LEFT_BORDER

						break
						
			if self.object_in_hand is None:
				# If bin 3 is not empty, then put it to an empty bin inside the reachable area(free the place for other agent to pass other pieces)
				if can_reach_ids[-1] is not None and can_get_3:
					for i in range(1, num_bins - 1):
						if can_reach_ids[i] is None:
							action = {"type": "pick", "obj_id": can_reach_ids[-1], "pos": np.array(self.id2pos[can_reach_ids[-1]]), "lift_up": True, "set_kinematic_state": True}
							self.object_in_hand = can_reach_ids[-1]
							self.pos_to_place = self.reachable_bins[i]
							if (i == 1 and self.is_clockwise) or (i == 2 and not self.is_clockwise):
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
							else:
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT

							break
							
			if self.object_in_hand is None:
				action = {"type": "wait"}

		return action

	def act(self, obs):
		# actions: pick object_id; place pos; wait
		
		if obs["rejected"]:
			self.pos_to_place = deepcopy(self.last_pos_to_place)
			self.object_in_hand = deepcopy(self.last_object_in_hand)
			self.place_type = deepcopy(self.last_place_type)
			self.wrong_piece_id = deepcopy(self.last_wrong_piece_id)
			self.place_wrong = deepcopy(self.last_place_wrong)

		self.last_pos_to_place = deepcopy(self.pos_to_place)
		self.last_object_in_hand = deepcopy(self.object_in_hand)
		self.last_place_type = deepcopy(self.place_type)
		self.last_wrong_piece_id = deepcopy(self.wrong_piece_id)
		self.last_place_wrong = deepcopy(self.place_wrong)
		
		action = None
		self.logger.debug(f"Test Agent {self.agent_id} acting")

		self.id2pos = {obj['id']: obj['pos'] for obj in obs["objects"]}
		self.id2name = {obj['id']: obj['name'] for obj in obs["objects"]}
		self.puzzle_id = None
		# print(self.puzzle_id2piece_id)

		# find a puzzle the agent is going to solve
		for puz_id in self.puzzle_id2piece_id.keys():
			# print(self.id2pos[puz_id], self.reachable_bins[2])
			if l2_dist(self.id2pos[puz_id], self.reachable_bins[2]) < 0.6:
				self.puzzle_id = puz_id
				break
		assert self.puzzle_id is not None
		self.puzzle_pos = self.id2pos[self.puzzle_id]
		
		# For every bin that that agent can reach, preprocess the piece_id in it
		can_reach_ids = [None for _ in range(len(self.reachable_bins))] 
		for i, pos in enumerate(self.reachable_bins):
			for puzzle_id in self.puzzle_ids:
				for piece in self.puzzle_id2piece_id[puzzle_id]:
					if l2_dist(self.id2pos[piece], pos) <= BIN_THRESHOLD:
						can_reach_ids[i] = piece
		
		num_bins = len(self.reachable_bins)

		free_bins = 0
		for i in range(1, len(can_reach_ids)):
			if can_reach_ids[i] is None:
				free_bins += 1

		possible_actions = [{"type": "wait", "prompt": "wait"}]

		if self.object_in_hand is None:
			for piece_id in can_reach_ids:
				if piece_id is not None:
					action = {"type": "pick", "obj_id": piece_id, "obj_name": self.id2name[piece_id], "pos": np.array(self.id2pos[piece_id]), "lift_up": True, "set_kinematic_state": True}
					action["prompt"] = self.action2prompt(action)
					possible_actions.append(action)
		else:
			for i, pos in enumerate(self.reachable_bins):
				if can_reach_ids[i] is None:
					if self.is_clockwise:
						if i == 0:
							place_type = PLACE_ON_THE_RIGHT_BORDER
						elif i == 1:
							place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						elif i == 2:
							place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
						elif i == 3:
							place_type = PLACE_ON_THE_LEFT_BORDER
						else:
							raise NotImplementedError
					else:
						if i == 0:
							place_type = PLACE_ON_THE_LEFT_BORDER
						elif i == 1:
							place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
						elif i == 2:
							place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						elif i == 3:
							place_type = PLACE_ON_THE_RIGHT_BORDER
						else:
							raise NotImplementedError

					action = {"type": "place", "obj_id": self.object_in_hand, "obj_name": self.id2name[self.object_in_hand], "pos": np.array(pos), "place_type": place_type, "set_kinematic_state": True}
					action["prompt"] = self.action2prompt(action)
					possible_actions.append(action)

			if self.object_in_hand in self.puzzle_id2piece_id[self.puzzle_id]:
				place_type = PLACE_INTO_PUZZLE

				for semantic_pos in self.annotation_dict[self.object_in_hand]:
					action = {"type": "place", "obj_id": self.object_in_hand, "obj_name": self.id2name[self.object_in_hand], "pos": np.array(self.puzzle_pos), "place_type": place_type, "set_kinematic_state": True}
					action["prompt"] = self.action2prompt(action, fix_semantic_pos=semantic_pos)
					possible_actions.append(action)

		# for piece_id in self.puzzle_id2piece_id[self.puzzle_id]:
		# 	piece_pos = self.id2pos[piece_id]
		# 	target_pos = self.puzzle_pos + convert_pos(int(self.agent_id), self.relative_position_dict[piece_id])
		# 	piece_pos[1] = target_pos[1]
		# 	print(self.agent_id, target_pos, piece_pos, l2_dist(target_pos, piece_pos), self.semantic_name_dict[piece_id])

		if self.wrong_piece_id is not None and free_bins > 0 and self.object_in_hand is None:
			for i in range(1, len(can_reach_ids)):
				if can_reach_ids[i] is None:
					action = {"type": "pick", "obj_id": self.wrong_piece_id, "obj_name": self.id2name[self.wrong_piece_id], 
								"pos": np.array(self.id2pos[self.wrong_piece_id]), "lift_up": True, "set_kinematic_state": True}
					
					self.object_in_hand = self.wrong_piece_id
					self.pos_to_place = np.array(self.reachable_bins[i])
					self.wrong_piece_id = None
					if self.is_clockwise:
						if i == 0:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER
						elif i == 1:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						elif i == 2:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
						elif i == 3:
							self.place_type = PLACE_ON_THE_LEFT_BORDER
						else:
							raise NotImplementedError
					else:
						if i == 0:
							self.place_type = PLACE_ON_THE_LEFT_BORDER
						elif i == 1:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
						elif i == 2:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						elif i == 3:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER
						else:
							raise NotImplementedError
					
					break

		elif self.object_in_hand is not None: # If picked a piece, then place it to pre-determined location
			if l2_dist(self.reachable_bins[1], self.pos_to_place) < BIN_THRESHOLD or l2_dist(self.reachable_bins[2], self.pos_to_place) < BIN_THRESHOLD:
				
				if self.random_dir:
					p = self.rng.choice([True, False])
					if p:
						if can_reach_ids[-1] is None:
							self.pos_to_place = self.reachable_bins[-1]
							if self.is_clockwise:
								self.place_type = PLACE_ON_THE_LEFT_BORDER
							else:
								self.place_type = PLACE_ON_THE_RIGHT_BORDER
					else:
						if can_reach_ids[0] is None:
							self.pos_to_place = self.reachable_bins[0]
							if self.is_clockwise:
								self.place_type = PLACE_ON_THE_RIGHT_BORDER
							else:
								self.place_type = PLACE_ON_THE_LEFT_BORDER

				elif self.nearest_dir:
					put_id = self.get_empty_bin(None, can_reach_ids, self.object_in_hand)
					if put_id is not None:
						self.pos_to_place = self.reachable_bins[put_id]
						if put_id == 0:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER
						elif put_id == 1:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						elif put_id == 2:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
						else:
							self.place_type = PLACE_ON_THE_LEFT_BORDER

				else:
					if can_reach_ids[-1] is None:
						self.pos_to_place = self.reachable_bins[-1]
						if self.is_clockwise:
							self.place_type = PLACE_ON_THE_LEFT_BORDER
						else:
							self.place_type = PLACE_ON_THE_RIGHT_BORDER

			put_id = None
			for i in range(num_bins):
				if l2_dist(self.pos_to_place, self.reachable_bins[i]) < BIN_THRESHOLD:
					put_id = i
					break

			if put_id is not None and can_reach_ids[put_id] is not None:
				flag = False
				for i in range(1, num_bins - 1):
					if can_reach_ids[i] is None:
						flag = True
						self.pos_to_place = self.reachable_bins[i]
						if (i == 1 and self.is_clockwise) or (i == 2 and not self.is_clockwise):
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
						else:
							self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT

						break

				if not flag:
					action = {"type": "wait"}
				else:
					self.pos_to_place[0] += self.rng.uniform(low=-0.02, high=0.02)
					self.pos_to_place[2] += self.rng.uniform(low=-0.02, high=0.02)
					action = {"type": "place", "pos": self.pos_to_place, "obj_id": self.object_in_hand, "place_type": self.place_type, "set_kinematic_state": True}
					
					self.object_in_hand = None
					self.pos_to_place = None
					self.place_type = None

			else:
				self.pos_to_place[0] += self.rng.uniform(low=-0.02, high=0.02)
				self.pos_to_place[2] += self.rng.uniform(low=-0.02, high=0.02)
				action = {"type": "place", "pos": self.pos_to_place, "obj_id": self.object_in_hand, "place_type": self.place_type, "set_kinematic_state": True}
				
				self.object_in_hand = None
				self.pos_to_place = None
				self.place_type = None

		else:
			in_other_slots = dict()
			for puzzle_id in self.puzzle_ids:
				for piece in self.puzzle_id2piece_id[puzzle_id]:
					if piece in can_reach_ids:
						in_other_slots[piece] = True
					else:
						in_other_slots[piece] = False

			# If a piece of puzzle that the agent needs can be reached, then pick it up and place it to the puzzle
			if self.admit_error_placing and self.rng.randint(0, 100) > 70 and self.wrong_piece_id is None:
				for i in range(len(can_reach_ids) - 1):
					if can_reach_ids[i] is not None:
						piece_id = can_reach_ids[i]
						piece_pos = self.id2pos[piece_id]
						target_pos = self.puzzle_pos + convert_pos(int(self.agent_id), self.relative_position_dict[piece_id])
						if (l2_dist(target_pos, piece_pos) > IN_PUZZLE_THRESHOLD or in_other_slots[piece_id]):
							action = {"type": "pick", "obj_id": piece_id, "obj_name": self.id2name[piece_id], "pos": np.array(piece_pos), "lift_up": True, "set_kinematic_state": True}
							self.object_in_hand = piece_id
							if piece_id not in self.puzzle_id2piece_id[self.puzzle_id]:
								self.pos_to_place = np.array(self.puzzle_pos)
								self.pos_to_place[1] += 0.1
								self.place_type = PLACE_INTO_PUZZLE
								self.place_wrong = True
								self.wrong_piece_id = piece_id
							else:
								self.pos_to_place = np.array(self.puzzle_pos)
								self.place_wrong = False
								self.place_type = PLACE_INTO_PUZZLE
								
							break

			else:
				for piece_id in self.puzzle_id2piece_id[self.puzzle_id]:
					piece_pos = self.id2pos[piece_id]
					target_pos = self.puzzle_pos + convert_pos(int(self.agent_id), self.relative_position_dict[piece_id])
					if self.within_reach(piece_pos) and (l2_dist(target_pos, piece_pos) > IN_PUZZLE_THRESHOLD or in_other_slots[piece_id]):
						action = {"type": "pick", "obj_id": piece_id, "obj_name": self.id2name[piece_id], "pos": np.array(piece_pos), "lift_up": True, "set_kinematic_state": True}
						
						self.object_in_hand = piece_id
						self.pos_to_place = np.array(self.puzzle_pos)
						self.place_type = PLACE_INTO_PUZZLE
						self.place_wrong = False
						break

			if self.object_in_hand is None:
				# Main idea: every agent pass the pieces in clock-wise order(to avoid conflict),
				# therefore, each agent receives pieces from bin indexed with 0 and pass the pieces to the bin indexed with num_bins - 1.
				# TODO: change to the shortest path to the agent that needs this piece


				# Check whether block in bin 0 can be acquired by other agent, if can, then don't pick it!
				can_get_0 = True 
				if can_reach_ids[0] is not None:
					piece_id = can_reach_ids[0]
					for puzzle_id in self.puzzle_ids:
						if piece_id in self.puzzle_id2piece_id[puzzle_id]:
							if l2_dist(self.id2pos[piece_id], self.id2pos[puzzle_id]) < 1.5:
								can_get_0 = False

				# Check whether block in bin 3 can be acquired by other agent, if can, then don't pick it!
				can_get_3 = True 
				if can_reach_ids[3] is not None:
					piece_id = can_reach_ids[3]
					for puzzle_id in self.puzzle_ids:
						if piece_id in self.puzzle_id2piece_id[puzzle_id]:
							if l2_dist(self.id2pos[piece_id], self.id2pos[puzzle_id]) < 1.5:
								can_get_3 = False
				
				if self.random_dir:
					action = self.pick_policy(can_reach_ids, num_bins, can_get_0, can_get_3, self.rng.choice([True, False]))
				elif self.nearest_dir:
					flag = False
					for pick_id in check_ord:
						if can_reach_ids[pick_id] is None or (not can_get_0 and pick_id == 0) or (not can_get_3 and pick_id == 3):
							continue

						put_id = self.get_empty_bin(pick_id, can_reach_ids, None)
						if put_id is not None and put_id != pick_id:
							action = {"type": "pick", "obj_id": can_reach_ids[pick_id], "obj_name": self.id2name[can_reach_ids[pick_id]], "pos": np.array(self.id2pos[can_reach_ids[pick_id]]), "lift_up": True, "set_kinematic_state": True}
							
							self.object_in_hand = can_reach_ids[pick_id]
							self.pos_to_place = self.reachable_bins[put_id]
							if put_id == 0:
								self.place_type = PLACE_ON_THE_RIGHT_BORDER
							elif put_id == 1:
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
							elif put_id == 2:
								self.place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
							else:
								self.place_type = PLACE_ON_THE_LEFT_BORDER

							flag = True
							break
					
					if not flag:
						action = {"type": "wait"}

				else:
					action = self.pick_policy(can_reach_ids, num_bins, can_get_0, can_get_3, True)

		if action["type"] != "wait":
			action["obj_name"] = self.id2name[action["obj_id"]]
			if int(self.agent_id) == 1 or int(self.agent_id) == 3:
				action["tag"] = True

		if action["type"] == "place" and action["place_type"] == PLACE_INTO_PUZZLE:
			for possible_action in possible_actions:
				if possible_action["type"] == "place" and possible_action["place_type"] == PLACE_INTO_PUZZLE \
					and possible_action["obj_id"] == action["obj_id"]:
					action["prompt"] = possible_action["prompt"]
					break
		else:
			action["prompt"] = self.action2prompt(action)

		action["prompt_proposer"] = self.convert_prompt_proposer([possible_action["prompt"] for possible_action in possible_actions], action["prompt"])
		self.last_action = action
		
		return action

	def within_reach(self, pos):
		return self.reachable_region[0] < pos[0] < self.reachable_region[2] and self.reachable_region[1] < pos[2] < \
			self.reachable_region[3]

	def check_goal(self, obs):
		completed = 0
		all = 0
		self.id2pos = {obj['id']: obj['pos'] for obj in obs["objects"]}
		for i, puzzle_id in enumerate(self.puzzle_ids):
			for j, piece in enumerate(self.puzzle_id2piece_id[puzzle_id]):
				piece_id = piece
				all += 1
				target_pos = self.id2pos[puzzle_id] + convert_pos(int(self.agent_id), self.relative_position_dict[piece_id])
				if l2_dist(self.id2pos[piece_id], target_pos) <= IN_PUZZLE_THRESHOLD:
					completed += 1

		return completed, all, completed == all
	
	def get_empty_bin(self, original_bin, can_reach_ids, piece_id):
		if original_bin is not None:
			piece_id = can_reach_ids[original_bin]
			assert piece_id is not None, f"There's no piece in the original bin!"

		for i, puzzle_id in enumerate(self.puzzle_ids):
			if piece_id in self.puzzle_id2piece_id[puzzle_id]:
				put_id = map_direction[int(self.agent_id)][i]
				if put_id == "random":
					if can_reach_ids[0] is not None or original_bin == 0:
						put_id = 3
					elif can_reach_ids[3] is not None or original_bin == 3:
						put_id = 0
					else:
						put_id = self.rng.choice([0, 3])
				
				if can_reach_ids[put_id] is None:
					return put_id

		return None

	def action2prompt(self, action, fix_semantic_pos=None):
		if action["type"] == "wait":
			return "wait"

		object_id = action["obj_id"]
		semantic_name = self.semantic_name_dict[object_id].replace('_', ' ')

		if action["type"] == 'pick':
			return f"pick up {semantic_name}"
		elif action["type"] == 'place':
			place_type = action["place_type"]
			if place_type == PLACE_INTO_PUZZLE:
				if fix_semantic_pos is not None:
					semantic_position = fix_semantic_pos.replace('_', ' ')

				else:
					if len(self.annotation_dict[object_id]) == 0:  # unexpected thing happens!
						semantic_position = "center"
					else:
						semantic_position = self.rng.choice(self.annotation_dict[object_id]).replace('_', ' ')

				return f"place {semantic_name} onto the {semantic_position} of the puzzle box"
			elif place_type == PLACE_ON_THE_LEFT_BORDER:
				return f"place {semantic_name} onto the left border of the reachable region"
			elif place_type == PLACE_ON_THE_RIGHT_BORDER:
				return f"place {semantic_name} onto the right border of the reachable region"
			elif place_type == PLACE_INSIDE_PRIVATE_AREA_LEFT:
				return f"place {semantic_name} onto the private region left to the puzzle box"
			elif place_type == PLACE_INSIDE_PRIVATE_AREA_RIGHT:
				return f"place {semantic_name} onto the private region right to the puzzle box"
			else:
				raise NotImplementedError(f"Unknown place type {place_type}")
		else:
			raise NotImplementedError(f"Unknown action type {action['type']}")

	def convert_prompt_proposer(self, possible_actions, chosen_action):
		assert chosen_action in possible_actions, f"Chosen action is not in possible actions"
		reachable_object_names = []
		for i, pos in enumerate(self.reachable_bins):
			for puzzle_id in self.puzzle_ids:
				for piece in self.puzzle_id2piece_id[puzzle_id]:
					if l2_dist(self.id2pos[piece], pos) <= BIN_THRESHOLD:
						reachable_object_names.append(self.semantic_name_dict[piece].replace('_', ' '))
		reachable_objects = ', '.join(reachable_object_names)
		if reachable_objects == '':
			reachable_objects = 'nothing'

		object_id = self.last_object_in_hand
		if object_id is None:
			semantic_name = 'nothing'
		else:
			semantic_name = self.semantic_name_dict[object_id].replace('_', ' ')

		return f"I'm holding {semantic_name}. I can see {reachable_objects} in my reachable region. My possible actions are: {', '.join(possible_actions)}. I choose to {chosen_action}."