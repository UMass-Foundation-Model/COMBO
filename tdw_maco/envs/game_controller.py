import numpy as np
import tdw

from envs.maco_controller import MacoController
from proc_gen.craftroom import CraftRoom
from components.avatar import *
from components.objects import *
from replicant import Replicant

BIN_THRESHOLD = 0.2
IN_PUZZLE_THRESHOLD = 0.1
agents_position = ["top", "left", "bottom", "right"]
number2word = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four"}
agent_adjacent_dict = {0: [1,3], 1: [0,2], 2: [1,3], 3: [2,0]}
agent_oppo_dict = {0: 2, 1: 3, 2: 0, 3: 1}
def l2_dist(pos1, pos2):
	return np.linalg.norm(np.array(pos1) - np.array(pos2))


def convert_actions_prompt(self, actions):
	agents_name = ["Alice", "Bob", "Charlie", "David"]
	prompt = ""
	for i in range(self.number_of_agents):
		if i == self.number_of_agents - 1:
			prompt += f"{agents_name[i]} {actions[str(i)]['prompt']}"
		else:
			prompt += f"{agents_name[i]} {actions[str(i)]['prompt']}\n"

	return prompt

def get_task_prompt(agent_id, recipe=None, agents_name=["Alice", "Bob", "Charlie", "David"]):
	num_agents = len(agents_name)
	agents_name_str = ", ".join(agents_name[:-1]) + ' and ' + agents_name[-1]
	return f"""{number2word[num_agents]} agents {agents_name_str} around a square table are cooperating together to solve a puzzle game. Each agent can only operate within the region of one table edge. The goal is to put all the pieces of the puzzle on the table into the correct puzzle box."""

def get_proposal_prompt(agent_id, recipe=None, agents_name=["Alice", "Bob", "Charlie", "David"]):
	agent_name = agents_name[agent_id]
	agent_pos = agents_position[agent_id]
	num_agents = len(agents_name)
	agents_name_str = ", ".join(agents_name[:-1]) + ' and ' + agents_name[-1]
	return f"""{number2word[num_agents]} agents {agents_name_str} around a square table are cooperating together to solve a puzzle game. Each agent can only operate within the region of one table edge. The goal is to put all the pieces of the puzzle on the table into the correct puzzle box.
As shown in the image, you're agent {agent_name} near the {agent_pos} edge of the table and you can only reach the table edge near you to pick and place objects. Given your reachability and the current state shown in the image, what are your possible actions?

Action Formats:
1) wait
2) pick up <obj>
3) place <obj> onto <loc>

For a formatted action, replace <obj> with the bi-colored piece name, such as "the green-black piece", "the aqua-yellow piece", "the brown-green piece", and <loc> with the location such as "the top edge of the puzzle box", "the bottom left corner of the puzzle box", "the left border of the reachable region", "the private region right to the puzzle box".
For example, the formatted action "place the aqua-yellow piece onto the left border of the reachable region" means you place the aqua-yellow piece onto the left border of the reachable region so that other agents may reach it.

Let's think step by step."""

def get_value_prompt(agent_id, recipe=None, agents_name=["Alice", "Bob", "Charlie", "David"]):
	num_agents = len(agents_name)
	agents_name_str = ", ".join(agents_name[:-1]) + ' and ' + agents_name[-1]
	return f"""{number2word[num_agents]} agents {agents_name_str} around a square table are cooperating together to solve a puzzle game. Each agent can only operate within the region of one table edge. The goal is to put all the pieces of the puzzle on the table into the correct puzzle box.
Describe the image regarding each item's position and the steps needed to finish the goal. 
Let's think step by step."""


''' augmented response: the green-black piece: in the adjacent agent's region, needs 4 more steps
the green-brown piece: in the adjacent agent's region, needs 4 more steps
the red-blue piece: in the opposite agent's region, needs 6 more steps
the yellow-purple piece: in the correct agent's region, needs 2 more steps
the yellow-black piece: in the correct agent's region, needs 2 more steps
the green-red piece: in the correct agent's region, needs 2 more steps
the aqua-yellow piece: in the correct agent's region, needs 2 more steps
the blue-brown piece: in the adjacent agent's region, needs 4 more steps
'''

def get_belief_prompt(agent_id, recipe=None, agents_name=["Alice", "Bob", "Charlie", "David"]):
	agent_name = agents_name[agent_id]
	agent_pos = agents_position[agent_id]
	num_agents = len(agents_name)
	agents_name_str = ", ".join(agents_name[:-1]) + ' and ' + agents_name[-1]
	other_agents = ", ".join([agent for i, agent in enumerate(agents_name) if i != agent_id])
	return f"""{number2word[num_agents]} agents {agents_name_str} around a square table are cooperating together to solve a puzzle game. Each agent can only operate within the region of one table edge. The goal is to put all the pieces of the puzzle on the table into the correct puzzle box.
Given {agent_name}'s observation history, what may {other_agents} do this time?"""
#
# Action Formats:
# 1) wait
# 2) pick up <obj>
# 3) place <obj> onto <loc>
#
# For a formatted action, replace <obj> with the bi-colored piece name, such as "the green-black piece", "the aqua-yellow piece", "the brown-green piece", and <loc> with the location such as "the top edge of the puzzle box", "the bottom left corner of the puzzle box", "the left border of the reachable region", "the private region right to the puzzle box".
# For example, the formatted action "place the aqua-yellow piece onto the left border of the reachable region" means you place the aqua-yellow piece onto the left border of the reachable region so that other agents may reach it.
#
# Let's think step by step."""

"""response:
Alice pick up the green-black piece
Bob pick up the green-brown piece
Charlie wait
"""
def convert_pos(id, pos):
	if id == 0:
		return np.array([-pos[0], pos[1], -pos[2]])
	elif id == 1:
		return np.array([pos[2], pos[1], -pos[0]])
	elif id == 2:
		return pos
	elif id == 3:
		return np.array([-pos[2], pos[1], pos[0]])

class GameController(MacoController):

	def __init__(self, number_of_agents=2, *args, **kwargs):
		super().__init__(avatars=[TopDownAvatar(position={"x": 0, "y": 6, "z": 0})], *args, **kwargs)
		self.obj = None
		self.puzzle_id2piece_id = None
		self.agents = []
		self.number_of_agents = number_of_agents
		self.table = None
		self.puzzles = None
		self.composite_object_manager = None
		self.reward = 0
		if self.number_of_agents == 4:
			self.agents_name = ["Alice", "Bob", "Charlie", "David"]
			self.agents_position = ["top", "left", "bottom", "right"]
		elif self.number_of_agents == 3:
			self.agents_name = ["Alice", "Bob", "Charlie"]
			self.agents_position = ["top", "left", "bottom"]
		elif self.number_of_agents == 2:
			self.agents_name = ["Alice", "Bob"]
			self.agents_position = ["top", "left"]

	def setup(self, seed: int = 42, number_of_agents: int = 2, is_test: bool = False):
		self.craftroom = CraftRoom()
		self.craftroom.create(self, number_of_agents, seed=seed, scale = {"x": 0.8, "y": 0.8, "z": 0.8}, is_test=is_test)
		around_table = self.craftroom.size / 2 + 0.45  # too large may cause the build to quit!
		on_table = self.craftroom.on_table
		height = self.craftroom.height
		gaps = self.craftroom.gaps
		self.puzzles = self.craftroom.puzzles
		self.puzzle_id2piece_id = self.craftroom.puzzle_id2piece_id
		self.relative_position_dict = self.craftroom.relative_position_dict
		self.semantic_name_dict = self.craftroom.semantic_name_dict
		self.semantic_name_2_id_dict = {v: k for k, v in self.semantic_name_dict.items()}
		self.annotation_dict = self.craftroom.annotation_dict
		self.rng = np.random.RandomState(seed)
		self.clockwise = self.rng.choice([True, False])

		if self.number_of_agents == 4:
			self.replicants = [Replicant(replicant_id=0,
										 state=self.state,
										 position=np.array([0, 0, around_table + 0.2]),
										 home_look_at=np.array([0, 0, -0.1]),
										 turn_left=np.array([0.3, 0, 0]),
										 rotation={"x": 0, "y": 180, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="woman_casual"),
								Replicant(replicant_id=1,
										 state=self.state,
										 position=np.array([-around_table - 0.2, 0, 0]),
										 home_look_at=np.array([0.1, 0, 0]),
										 turn_left=np.array([0, 0.3, 0]),
										 rotation={"x": 0, "y": 90, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="man_casual"),
								Replicant(replicant_id=2,
										 state=self.state,
										 position=np.array([0, 0, -around_table - 0.2]),
										 home_look_at=np.array([0, 0, 0.1]),
										 turn_left=np.array([-0.3, 0, 0]),
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="fireman"),
								Replicant(replicant_id=3,
										 state=self.state,
										 position=np.array([around_table + 0.2, 0, 0]),
										 home_look_at=np.array([-0.1, 0, 0]),
										 turn_left=np.array([0, -0.3, 0]),
										 rotation={"x": 0, "y": -90, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="replicant_0")]

			self.bots = []
			self.reachable_region = [[-around_table, on_table - 0.2, around_table, around_table],
									 [-around_table, -around_table, -on_table + 0.2, around_table],
									 [-around_table, -around_table, around_table, -on_table + 0.2],
									 [on_table - 0.2, -around_table, around_table, around_table]] # [x1, z1, x2, z2], [y1, y2]

			# print(self.reachable_region)

			self.reachable_bins = [[[gaps[i], height + 0.1, on_table] for i in range(len(gaps))],
						            [[-on_table, height + 0.1, gaps[i]] for i in range(len(gaps))],
									[[gaps[i], height + 0.1, -on_table] for i in range(len(gaps) - 1, -1, -1)],
									[[on_table, height + 0.1, gaps[i]] for i in range(len(gaps) - 1, -1, -1)]]

		elif self.number_of_agents == 3:
			self.replicants = [Replicant(replicant_id=0,
										 state=self.state,
										 position=np.array([0, 0, around_table + 0.2]),
										 home_look_at=np.array([0, 0, 0]),
										 turn_left=np.array([0.3, 0, 0]),
										 rotation={"x": 0, "y": 180, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="woman_casual"),
								Replicant(replicant_id=1,
										 state=self.state,
										 position=np.array([-around_table - 0.2, 0, 0]),
										 home_look_at=np.array([0, 0, 0]),
										 turn_left=np.array([0, 0.3, 0]),
										 rotation={"x": 0, "y": 90, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="man_casual"),
								Replicant(replicant_id=2,
										 state=self.state,
										 position=np.array([0, 0, -around_table - 0.2]),
										 home_look_at=np.array([0, 0, 0]),
										 turn_left=np.array([-0.3, 0, 0]),
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="fireman")]

			self.bots = []
			self.reachable_region = [[-around_table, on_table - 0.2, around_table, around_table],
									[-around_table, -around_table, -on_table + 0.2, around_table],
									 [-around_table, -around_table, around_table, -on_table + 0.2]] # [x1, z1, x2, z2], [y1, y2]

			# print(self.reachable_region)

			self.reachable_bins = [[[gaps[i], height + 0.1, on_table] for i in range(len(gaps))],
						            [[-on_table, height + 0.1, gaps[i]] for i in range(len(gaps))],
									[[gaps[i], height + 0.1, -on_table] for i in range(len(gaps) - 1, -1, -1)]]

		elif self.number_of_agents == 2:
			self.replicants = [Replicant(replicant_id=0,
										 state=self.state,
										 position=np.array([0, 0, around_table + 0.2]),
										 home_look_at=np.array([0, 0, 0]),
										 turn_left=np.array([0.3, 0, 0]),
										 rotation={"x": 0, "y": 180, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="woman_casual"),
								Replicant(replicant_id=1,
										 state=self.state,
										 position=np.array([-around_table - 0.2, 0, 0]),
										 home_look_at=np.array([0, 0, 0]),
										 turn_left=np.array([0, 0.3, 0]),
										 rotation={"x": 0, "y": 90, "z": 0},
										 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
										 # frequency.always give no initial frame obs!!!
										 target_framerate=self.target_framerate,
										 enable_collision_detection=self.enable_collision_detection,
										 name="man_casual")]

			self.bots = []
			self.reachable_region = [[-around_table, on_table - 0.2, around_table, around_table],
									 [-around_table, -around_table, -on_table + 0.2, around_table]] # [x1, z1, x2, z2], [y1, y2]

			# print(self.reachable_region)

			self.reachable_bins = [[[gaps[i], height + 0.1, on_table] for i in range(len(gaps))],
									[[-on_table, height + 0.1, gaps[i]] for i in range(len(gaps))]]

		self.agents = self.replicants + self.bots
		self._init_agents()

		for agent in self.agents:
			for i, puzzle in enumerate(self.puzzles):
				for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
					agent.collision_detection.exclude_objects.append(piece)

		# replicant.pick_up(puzzles[0].id)
		# while replicant.action.status == RepActionStatus.ongoing:
		# 	c.communicate([])
		# 	print("replicant picking up puzzle")
		# 	print(replicant.action.status)

		self._render()
		self.reward = 0
		info = {"agents_name": self.agents_name,
				"reachable_region": self.reachable_region,
				"reachable_bins": self.reachable_bins,
				"puzzle_id2piece_id": self.puzzle_id2piece_id,
				"relative_position_dict": self.relative_position_dict,
				"semantic_name_dict": self.semantic_name_dict,
				"annotation_dict": self.annotation_dict,
				"is_clockwise": self.clockwise,
				"seed": seed}

		return info

	def get_objects(self):
		self.obj = []

		for i, puzzle in enumerate(self.puzzles):
			self.obj.append({
					'id': puzzle.id,
					'name': puzzle.name,
					'pos': puzzle.transform.position,
					"upbound_pos": puzzle.bound.top,
				})
			for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
				piece_id = piece
				self.obj.append({
					'id': piece_id,
					'name': self.object_manager.objects_static[piece_id].name,
					'pos': self.object_manager.transforms[piece_id].position,
					"upbound_pos": None
				})

		return self.obj

	def parse_text_action(self, action, agent_id):
		try:
			if action == "wait":
				return {"type": "wait"}
			elif action.startswith("pick up"):
				obj_semantic_name = action.split("pick up ")[1].replace(" ", "_")
				obj_id = self.semantic_name_2_id_dict[obj_semantic_name]
				obj_pos = None
				for obj in self.obj:
					if obj["id"] == obj_id:
						obj_pos = obj["pos"]
						break
				assert obj_pos is not None, f"obj_pos is None for {obj_semantic_name} {obj_id}"
				if int(agent_id) == 0 or int(agent_id) == 2:
					action = {"type": "pick", "obj_id": obj_id, "pos": np.array(obj_pos), "lift_up": True, "set_kinematic_state": True}
				else:
					action = {"type": "pick", "obj_id": obj_id, "pos": np.array(obj_pos), "lift_up": True, "tag": True, "set_kinematic_state": True}
				return action
			elif action.startswith("place"):
				PLACE_INTO_PUZZLE = 1
				PLACE_ON_THE_LEFT_BORDER = 2
				PLACE_ON_THE_RIGHT_BORDER = 3
				PLACE_INSIDE_PRIVATE_AREA_LEFT = 4
				PLACE_INSIDE_PRIVATE_AREA_RIGHT = 5

				obj_semantic_name = action.split("place ")[1].split(" onto ")[0].replace(" ", "_")
				loc = action.split(" onto ")[1]
				obj_id = self.semantic_name_2_id_dict[obj_semantic_name]
				if "of the puzzle box" in loc:
					place_type = PLACE_INTO_PUZZLE
					locc = loc.split(" of the puzzle box")[0]
					locc = locc.split("the ")[1]
					pos_to_place = self.puzzles[agent_id].transform.position
				elif "of the reachable region" in loc:
					locc = loc.split(" border of the reachable region")[0]
					locc = locc.split("the ")[1]
					place_type = PLACE_ON_THE_LEFT_BORDER if locc == "left" else PLACE_ON_THE_RIGHT_BORDER
					pos_to_place = self.reachable_bins[agent_id][-1 if locc == "left" else 0]
				elif "the private region left to the puzzle box" in loc:
					place_type = PLACE_INSIDE_PRIVATE_AREA_LEFT
					# loc = loc.split(" of the private area")[0]
					# loc = loc.split("the ")[1]
					pos_to_place = self.reachable_bins[agent_id][2]
				elif "the private region right to the puzzle box" in loc:
					place_type = PLACE_INSIDE_PRIVATE_AREA_RIGHT
					# loc = loc.split(" of the private area")[0]
					# loc = loc.split("the ")[1]
					pos_to_place = self.reachable_bins[agent_id][1]
				else:
					raise NotImplementedError(f"loc {loc} not implemented")
				if int(agent_id) == 0 or int(agent_id) == 2:
					action = {"type": "place", "pos": pos_to_place, "obj_id": obj_id, "place_type": place_type, "set_kinematic_state": True}
				else:
					action = {"type": "place", "pos": pos_to_place, "obj_id": obj_id, "place_type": place_type, "tag": True, "set_kinematic_state": True}
				return action
		except:
			return {"type": "wait"}

	def check_goal(self):
		completed = 0
		all = 0
		for i, puzzle in enumerate(self.puzzles):
			for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
				piece_id = piece
				target_pos = self.object_manager.transforms[puzzle.id].position + convert_pos(i, self.relative_position_dict[piece_id])
				all += 1
				if l2_dist(self.object_manager.transforms[piece_id].position, target_pos) <= IN_PUZZLE_THRESHOLD:
					completed += 1

		return completed, all, completed == all

	def convert_actions_prompt(self, actions):
		prompt = ""
		for i in range(self.number_of_agents):
			if i == self.number_of_agents - 1:
				prompt += f"{self.agents_name[i]} {actions[str(i)]['prompt']}"
			else:
				prompt += f"{self.agents_name[i]} {actions[str(i)]['prompt']}\n"

		return prompt

	def get_reward(self, info):
		reward = 0
		prompt_value = ""
		held_objects = info["held_objects"]
		for i, puzzle in enumerate(self.puzzles): # ith puzzle belongs to agent i
			for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
				piece_id = piece
				piece_semantic_name = self.semantic_name_dict[piece_id]
				if piece_id == held_objects[i]:
					prompt_value += f"{piece_semantic_name}: in the correct agent's hand, needs 1 more step\n"
					reward += 1
					continue
				if (agent_adjacent_dict[i][0] < self.number_of_agents and piece_id == held_objects[agent_adjacent_dict[i][0]]) or (agent_adjacent_dict[i][1] < self.number_of_agents and piece_id == held_objects[agent_adjacent_dict[i][1]]):
					prompt_value += f"{piece_semantic_name}: in the adjacent agent's hand, needs 3 more steps\n"
					reward += 3
					continue
				if agent_oppo_dict[i] < self.number_of_agents and piece_id == held_objects[agent_oppo_dict[i]]:
					prompt_value += f"{piece_semantic_name}: in the opposite agent's hand, needs 5 more steps\n"
					reward += 5
					continue
				target_pos = self.object_manager.transforms[puzzle.id].position + convert_pos(i, self.relative_position_dict[piece_id])
				dist = l2_dist(self.object_manager.transforms[piece_id].position, target_pos)
				if dist <= IN_PUZZLE_THRESHOLD:
					# in the correct box
					prompt_value += f"{piece_semantic_name}: in the correct box, needs 0 more steps\n"
					reward += 0
					continue

				least_reward = 100
				for agent_id, pos_list in enumerate(self.reachable_bins):
					if any([l2_dist(self.object_manager.transforms[piece_id].position, pos) <= BIN_THRESHOLD for pos in pos_list]):
						if agent_id == i:
							least_reward = min(least_reward, 2)
						elif agent_id in agent_adjacent_dict[i]:
							least_reward = min(least_reward, 4)
						else:
							least_reward = min(least_reward, 6)
						
				if least_reward == 2:
					prompt_value += f"{piece_semantic_name}: in the correct agent's region, needs 2 more steps\n"
					reward += 2
				elif least_reward == 4:
					prompt_value += f"{piece_semantic_name}: in the adjacent agent's region, needs 4 more steps\n"
					reward += 4
				elif least_reward == 6:
					prompt_value += f"{piece_semantic_name}: in the opposite agent's region, needs 6 more steps\n"
					reward += 6
				else:
					raise AssertionError

				
		reward = self.reward - reward
		self.reward = reward
		return reward, prompt_value