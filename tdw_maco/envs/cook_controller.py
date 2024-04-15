import tdw

from proc_gen.kitchen import Kitchen
from envs.maco_controller import MacoController
from components.avatar import *
from replicant import Replicant
from components.objects import *
import numpy as np

THRESHOLD_0 = -0.1
THRESHOLD_1 = -1.1
BIN_DIST_THRESHOLD = 0.1
BOARD_DIST_THRESHOLD = 0.2
PLATE_DIST_THRESHOLD = 0.25
WHOLE_TO_SLICE = {"whole_cheese": "cheese_slice", "whole_tomato": "tomato_slice", "whole_onion": "onion_slice"}
SLICE_TO_WHOLE = {v: k for k, v in WHOLE_TO_SLICE.items()}
agents_name = ["Alice", "Bob"]
agents_position = ["left", "bottom"]
def l2_dist(pos1, pos2):
    if isinstance(pos1, dict):
        pos1 = np.array([pos1["x"], pos1["y"], pos1["z"]])
    if isinstance(pos2, dict):
        pos2 = np.array([pos2["x"], pos2["y"], pos2["z"]])

    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def convert_actions_prompt(actions):
    return f"""{agents_name[0]} {actions["0"]['prompt']}
{agents_name[1]} {actions["1"]['prompt']}"""

def get_task_prompt(agent_id, recipe, agents_name= ["Alice", "Bob"]):
    recipe_strs = []
    for recipe_single in recipe:
        recipe_str = ", ".join(recipe_single)
        if recipe_str.startswith("burger_bottom"):
            recipe_str = "burger: " + recipe_str
        else:
            recipe_str = "sandwich: " + recipe_str
        recipe_strs.append(recipe_str)
    agent_name = agents_name[agent_id]
    return f"""Two agents Alice and Bob are cooperating together to cook at a kitchen counter. Each agent can only operate within the region of one counter edge. The goal is to make a burger and a sandwich. Food items must be stacked on the plate following this order:
{recipe_strs[0]}
{recipe_strs[1]}"""

def get_proposal_prompt(agent_id, recipe, agents_name= ["Alice", "Bob"]):
    recipe_strs = []
    for recipe_single in recipe:
        recipe_str = ", ".join(recipe_single)
        if recipe_str.startswith("burger_bottom"):
            recipe_str = "burger: " + recipe_str
        else:
            recipe_str = "sandwich: " + recipe_str
        recipe_strs.append(recipe_str)
    agent_name = agents_name[agent_id]
    agent_pos = agents_position[agent_id]
    return f"""Two agents Alice and Bob are cooperating together to cook at a kitchen counter. Each agent can only operate within the region of one counter edge. The goal is to make a burger and a sandwich. Food items must be stacked on the plate following this order:
{recipe_strs[0]}
{recipe_strs[1]}
Notice some food items may need to be cut on the cutting board first to get the needed one in the recipe. 
As shown in the image, you're agent {agent_name} near the {agent_pos} edge of the cabinet making the {recipe_strs[agent_id].split(":")[0]} and you can only reach the counter edge near you to pick and place objects, or cut objects on the cutting board. Given your reachability and the current state shown in the image, what are your possible actions?

Action Formats:
1) wait
2) pick up <obj>
3) place <obj> onto <loc>
4) cut <obj>

For a formatted action, replace <obj> with the food item name, such as "the pickle_slice", "the whole_onion", "the burger_bottom", and <loc> with the location such as "the cutting board", "the plate", "the top left corner of the private region".
For example, the formatted action "place the whole_cheese onto the cutting board" means you place the whole_cheese onto the cutting board so that either agent can cut it to cheese_slice or the other agent may reach it.

Let's think step by step."""

def get_value_prompt(agent_id, recipe, agents_name= ["Alice", "Bob"]):
    recipe_strs = []
    for recipe_single in recipe:
        recipe_str = ", ".join(recipe_single)
        if recipe_str.startswith("burger_bottom"):
            recipe_str = "burger: " + recipe_str
        else:
            recipe_str = "sandwich: " + recipe_str
        recipe_strs.append(recipe_str)
    return f"""Two agents Alice and Bob are cooperating together to cook at a kitchen counter. Each agent can only operate within the region of one counter edge. The goal is to make a burger and a sandwich. Food items must be stacked on the plate following this order:
{recipe_strs[0]}
{recipe_strs[1]}
Notice some food items may need to be cut on the cutting board first to get the needed one in the recipe.
Describe the image regarding each item's position and the steps needed to finish the goal.
Let's think step by step."""


''' augmented response: slice_of_bread: in the correct agent's region, needs 2 more steps
slice_of_cheese: needs to be cut, the block_of_cheese is in the other agent's region, needs 5 more steps
pickle_slice: in the other agent's region, needs 4 more steps
slice_of_bread: in the correct agent's region, needs 2 more steps
burger_bottom: in the other agent's region, needs 4 more steps
ham_slice: in the correct agent's region, needs 2 more steps
lettuce: in the other agent's region, needs 4 more steps
burger_top: in the correct agent's region, needs 2 more steps
'''

def get_belief_prompt(agent_id, recipe, agents_name= ["Alice", "Bob"]):
    recipe_strs = []
    for recipe_single in recipe:
        recipe_str = ", ".join(recipe_single)
        if recipe_str.startswith("burger_bottom"):
            recipe_str = "burger: " + recipe_str
        else:
            recipe_str = "sandwich: " + recipe_str
        recipe_strs.append(recipe_str)
    agent_name = agents_name[agent_id]
    other_agents = ", ".join([agent for i, agent in enumerate(agents_name) if i != agent_id])
    return f"""Two agents Alice and Bob are cooperating together to cook at a kitchen counter. Each agent can only operate within the region of one counter edge. The goal is to make a burger and a sandwich. Food items must be stacked on the plate following this order:
{recipe_strs[0]}
{recipe_strs[1]}
Notice some food items may need to be cut on the cutting board first to get the needed one in the recipe.
Given {agent_name}'s observation history, what may {other_agents} do this time?"""
#
# Action Formats:
# 1) wait
# 2) pick up <obj>
# 3) place <obj> onto <loc>
# 4) cut <obj>
#
# For a formatted action, replace <obj> with the food item name, such as "the pickle_slice", "the whole_onion", "the burger_bottom", and <loc> with the location such as "the cutting board", "the plate", "the top left corner of the private region".
# For example, the formatted action "place the whole_cheese onto the cutting board" means you place the whole_cheese onto the cutting board so that either agent can cut it to cheese_slice or the other agent may reach it.
#
# Let's think step by step."""

"""response:
Alice pick up the brown_bread_slice
Bob pick up the whole_cheese
"""

PLACE_ON_THE_CUTTING_BOARD = 11
PLACE_ON_THE_PLATE = 12
PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT = 13
PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT = 14
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT = 15
PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT = 16


class CookController(MacoController):

    def __init__(self, *args, **kwargs):
        super().__init__(avatars=[CookTopDownAvatar()], *args, **kwargs)
        self.random_state = None
        self.private_region = None
        self.agents = None
        self.kitchen = None
        self.reachable_region = None
        self.reachable_bins = None
        self.agents_name = ["Alice", "Bob"]
        self.reward = 0

    def setup(self, seed: int = 42, number_of_agents: int = 2, is_test: bool = False):
        self.kitchen = Kitchen()
        self.kitchen.create(self, seed=seed, is_test=is_test)

        self.reachable_bins = [self.kitchen.bins[0:4], self.kitchen.bins[4:]]
        for i in range(len(self.reachable_bins)):
            for j in range(len(self.reachable_bins[i])):
                self.reachable_bins[i][j] = [self.reachable_bins[i][j]["x"], self.reachable_bins[i][j]["y"], self.reachable_bins[i][j]["z"]]

        agent0_z = self.kitchen.pos_z - 0.7
        agent1_x = self.kitchen.pos_x + 0.7

        self.agents = [Replicant(replicant_id=0,
                                 state=self.state,
                                 position=np.array([-0.05, 0, agent0_z]),
                                 home_look_at=np.array([0.05, 0, agent0_z + 1.7]),
                                 rotation={"x": 0, "y": 0, "z": 0},
                                 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
                                 target_framerate=self.target_framerate,
                                 enable_collision_detection=self.enable_collision_detection,
                                 name="woman_casual"),
                       Replicant(replicant_id=1,
                                 state=self.state,
                                 position=np.array([agent1_x, 0, 0.8]),
                                 home_look_at=np.array([agent1_x - 1.7, 0, 0.7]),
                                 rotation={"x": 0, "y": 270, "z": 0},
                                 image_frequency=tdw.replicant.image_frequency.ImageFrequency.always,
                                 target_framerate=self.target_framerate,
                                 enable_collision_detection=self.enable_collision_detection,
                                 name="man_casual")]

        self.reachable_region = [[-2, self.kitchen.pos_z - 0.2, 2, self.kitchen.pos_z + 0.2],
                                 [self.kitchen.pos_x - 0.2, -2, self.kitchen.pos_x + 0.2, 2]] # [x1, z1, x2, z2]

        size = self.kitchen.tables[0].original_size[0]
        self.private_region = [[self.kitchen.pos_x - 2 * size - 0.3, self.kitchen.pos_z - 0.2, self.kitchen.pos_x - size + 0.3, self.kitchen.pos_z + 0.2],
                               [self.kitchen.pos_x - 0.2, self.kitchen.pos_z + size - 0.3, self.kitchen.pos_x + 0.2, self.kitchen.pos_z + 2 * size + 0.3]] # [x1, z1, x2, z2]
        # bench = WoodBench(self.object_manager, {"x": -1.4, "y": 0, "z": 0.3}, scale={"x": 0.3, "y": 1.3, "z": 0.3})
        # self.add_ons.append(bench)
        self._init_agents()

        self.random_state = np.random.RandomState(seed)
        self.recipe = self.kitchen.recipe
        self.knife_id = self.kitchen.knife.id
        self.knife_home_pos = self.kitchen.knife.transform.position
        self.objects = self.kitchen.food + [self.kitchen.cutting_board]

        for agent in self.agents:
            for i, obj in enumerate(self.objects):
                agent.collision_detection.exclude_objects.append(obj.id)


        self._render()
        self.reward = 0
        info = {"agents_name": self.agents_name,
                "recipe": self.recipe,
                "reachable_region": self.reachable_region,
                "private_region": self.private_region,
                "dest_plates": [self.kitchen.plates[i].bound.top for i in range(2)],
                "reachable_bins": self.reachable_bins,
                "seed": seed}

        return info

    def get_objects(self):
        self.obj = []
        for i, obj in enumerate(self.objects):
            self.obj.append({
                'id': obj.id,
                'name': obj.name,
                'pos': obj.transform.position,
                "upbound_pos": obj.bound.top
            })

        return self.obj

    def get_object_by_name(self, name) -> Object:
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def prompt_to_pos(self, place_type, agent_id):
        if place_type == PLACE_ON_THE_CUTTING_BOARD:
            return self.get_object_by_name("wood_board").bound.top
        elif place_type == PLACE_ON_THE_PLATE:
            num = [0, 0]
            (num[0], num[1]), _, _ = self.check_goal()
            if num[agent_id] == 0:
                return self.kitchen.plates[agent_id].bound.top
            else:
                return self.get_object_by_name(self.recipe[agent_id][num[agent_id] - 1]).bound.top
        else:
            return self.reachable_bins[agent_id][place_type - PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT]

    def parse_text_action(self, action, agent_id):
        try:
            if action == "wait":
                return {"type": "wait"}
            elif "pick up the " in action:
                obj_name = action.split("pick up the ")[1]
                obj = None
                for o in self.obj:
                    if o["name"] == obj_name:
                        obj = o
                        break
                if obj is None:
                    return {"type": "wait", "reject": True}
                assert obj is not None, f"obj is None for {obj_name}"

                if int(agent_id) == 1:
                    return {"type": "pick", "obj_id": obj["id"], "obj_name": obj["name"], "pos": obj["pos"], "tag": True, "set_kinematic_state": True, "lift_up": True, "offset": 0.05}
                else:
                    return {"type": "pick", "obj_id": obj["id"], "obj_name": obj["name"], "pos": obj["pos"], "set_kinematic_state": True, "lift_up": True, "offset": 0.05}

            elif "cut the " in action:
                obj_name = action.split("cut the ")[1]
                obj = None
                for o in self.obj:
                    if o["name"] == obj_name:
                        obj = o
                        break
                if obj is None:
                    return {"type": "wait", "reject": True}
                assert obj is not None, f"obj is None for {obj_name}"

                if int(agent_id) == 1:
                    return {"type": "cut", "obj_id": obj["id"], "obj_name": obj["name"], "pos": obj["pos"], "tag": True}
                else:
                    return {"type": "cut", "obj_id": obj["id"], "obj_name": obj["name"], "pos": obj["pos"]}

            elif "place the " in action:
                obj_name, loc_name = action.split("place the ")[1].split(" onto the ")

                obj = None
                for o in self.obj:
                    if o["name"] == obj_name:
                        obj = o
                        break
                if obj is None:
                    return {"type": "wait", "reject": True}
                assert obj is not None, f"obj is None for {obj_name}"

                place_type = None
                if "plate" in loc_name:
                    place_type = PLACE_ON_THE_PLATE
                elif "cutting board" in loc_name:
                    place_type = PLACE_ON_THE_CUTTING_BOARD
                elif "top left" in loc_name:
                    place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_LEFT
                elif "top right" in loc_name:
                    place_type = PLACE_IN_THE_PRIVATE_REGION_TOP_RIGHT
                elif "bottom left" in loc_name:
                    place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_LEFT
                elif "bottom right" in loc_name:
                    place_type = PLACE_IN_THE_PRIVATE_REGION_BOTTOM_RIGHT
                else:
                    raise AssertionError(f"loc_name is not recognized: {loc_name}")
                
                place_pos = self.prompt_to_pos(place_type, int(agent_id))
                if isinstance(place_pos, dict):
                    place_pos = np.array([place_pos["x"], place_pos["y"], place_pos["z"]])

                if int(agent_id) == 1:
                    return {"type": "place", "obj_id": obj["id"], "obj_name": obj["name"], "pos": place_pos, "tag": True, "place_type": place_type, "set_kinematic_state": True}
                else:
                    return {"type": "place", "obj_id": obj["id"], "obj_name": obj["name"], "pos": place_pos, "place_type": place_type, "set_kinematic_state": True}
        except:
            return {"type": "wait", "reject": True}

    def convert_actions_prompt(self, actions):
        return f"""{self.agents_name[0]} {actions["0"]['prompt']}
{self.agents_name[1]} {actions["1"]['prompt']}"""

    def check_goal(self):
        plate1_pos = self.kitchen.plates[0].bound.top
        num1 = len(self.recipe[0])
        last_height = -5
        for i in range(len(self.recipe[0])):
            name = self.recipe[0][i]
            objs = []
            for obj in self.objects:
                if obj.name == name and l2_dist(obj.transform.position, plate1_pos) < PLATE_DIST_THRESHOLD and obj.transform.position[1] > last_height:
                    objs.append(obj)

            if len(objs) == 0:
                num1 = i
                break

            if len(objs) > 1:
                objs = sorted(objs, key=lambda obj: obj.transform.position[1])

            last_height = objs[0].transform.position[1]

        last_height = -5
        plate2_pos = self.kitchen.plates[1].bound.top
        num2 = len(self.recipe[1])
        for i in range(len(self.recipe[1])):
            name = self.recipe[1][i]
            objs = []
            for obj in self.objects:
                if obj.name == name and l2_dist(obj.transform.position, plate2_pos) < PLATE_DIST_THRESHOLD and obj.transform.position[1] > last_height:
                    objs.append(obj)

            if len(objs) == 0:
                num2 = i
                break

            if len(objs) > 1:
                objs = sorted(objs, key=lambda obj: obj.transform.position[1])

            last_height = objs[0].transform.position[1]

        return (num1, num2), (len(self.recipe[0]), len(self.recipe[1])), (num1 == len(self.recipe[0]) and num2 == len(self.recipe[1]))

    def get_reward(self, info):
        prompt_value = ""
        held_objects = info["held_objects"]
        reward = 0
        cutting_board_pos = self.kitchen.cutting_board.transform.position

        num = [0, 0]
        ln = [0, 0]
        (num[0], num[1]), (ln[0], ln[1]), done = self.check_goal()
        for agent_id, agent_recipe in enumerate(self.recipe):
            another_agent_id = 1 if agent_id == 0 else 0
            for recipe_obj in agent_recipe[:num[agent_id]]:
                prompt_value += recipe_obj + f": on the plate, needs 0 more steps\n"
            for recipe_obj in agent_recipe[num[agent_id]:]:
                _obj = self.get_object_by_name(recipe_obj) # need to deal with the two breads
                if _obj is None:
                    # must be cut first
                    assert recipe_obj in SLICE_TO_WHOLE.keys(), f"{recipe_obj} is not in SLICE_TO_WHOLE"
                    whole_obj = SLICE_TO_WHOLE[recipe_obj]
                    _obj = self.get_object_by_name(whole_obj)
                    obj_id = _obj.id
                    if l2_dist(self.object_manager.transforms[obj_id].position, cutting_board_pos) <= BOARD_DIST_THRESHOLD:
                        prompt_value += recipe_obj + f": needs to be cut, the {whole_obj} is on the cutting board, needs 3 more steps\n"
                        reward += 3
                    elif obj_id == held_objects[agent_id]:
                        prompt_value += recipe_obj + f": needs to be cut, the {whole_obj} is in the correct agent's hand, needs 4 more steps\n"
                        reward += 4
                    elif obj_id == held_objects[another_agent_id]:
                        prompt_value += recipe_obj + f": needs to be cut, the {whole_obj} is in the other agent's hand, needs 4 more steps\n"
                        reward += 4
                    else:
                        for reachable_agent_id, pos_list in enumerate(self.reachable_bins):
                            if any([l2_dist(self.object_manager.transforms[obj_id].position, pos) <= BIN_DIST_THRESHOLD for pos in pos_list]):
                                if reachable_agent_id == agent_id:
                                    prompt_value += recipe_obj + f": needs to be cut, the {whole_obj} is in the correct agent's region, needs 5 more steps\n"
                                    reward += 5
                                else:
                                    prompt_value += recipe_obj + f": needs to be cut, the {whole_obj} is in the other agent's region, needs 5 more steps\n"
                                    reward += 5

                                break
                    continue

                obj_id = _obj.id
                if obj_id == held_objects[agent_id]:
                    prompt_value += f"{recipe_obj}: in the correct agent's hand, needs 1 more step\n"
                    reward += 1
                    continue
                elif obj_id == held_objects[another_agent_id]:
                    prompt_value += f"{recipe_obj}: in the other agent's hand, needs 3 more steps\n"
                    reward += 3
                    continue
                elif l2_dist(self.object_manager.transforms[obj_id].position, cutting_board_pos) <= BOARD_DIST_THRESHOLD:
                    prompt_value += recipe_obj + f": on the cutting_board, needs 2 more steps\n"
                    reward += 2
                    continue
                else:
                    for reachable_agent_id, pos_list in enumerate(self.reachable_bins):
                        if any([l2_dist(self.object_manager.transforms[obj_id].position, pos) <= BIN_DIST_THRESHOLD for pos in pos_list]):
                            if reachable_agent_id == agent_id:
                                prompt_value += recipe_obj + f": in the correct agent's region, needs 2 more steps\n"
                                reward += 2
                            else:
                                prompt_value += recipe_obj + f": in the other agent's region, needs 4 more steps\n"
                                reward += 4

                            break
                    
        reward = self.reward - reward
        self.reward = reward
        return reward, prompt_value