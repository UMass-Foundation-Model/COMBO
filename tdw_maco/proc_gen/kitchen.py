from typing import List

from envs.maco_controller import MacoController
from components.objects import *
from tdw.replicant.collision_detection import CollisionDetection
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
import numpy as np
import copy

# RECIPES = {
#     "ham-tomato": [["burger_bottom", "ham_slice", "burger_top"], ["bread_slice", "tomato_slice", "onion_slice", "bread_slice"]],
#     "patty-lettuce": [["burger_bottom", "hamburger_patty", "cheese_slice", "burger_top"], ["bread_slice", "lettuce", "bread_slice"]],
#     "ham-pickle": [["burger_bottom", "ham_slice", "lettuce", "burger_top"], ["bread_slice", "cheese_slice", "pickle_slice", "bread_slice"]],
#     "patty-onion": [["burger_bottom", "cheese_slice", "hamburger_patty", "tomato_slice", "burger_top"], ["bread_slice", "onion_slice", "bread_slice"]],
#     "onion-tomato": [["burger_bottom", "onion_slice", "burger_top"], ["bread_slice", "tomato_slice", "cheese_slice", "lettuce", "bread_slice"]],
# }

WHOLE_TO_SLICE = {"whole_cheese": "cheese_slice", "whole_tomato": "tomato_slice", "whole_onion": "onion_slice"}
SLICE_TO_WHOLE = {v: k for k, v in WHOLE_TO_SLICE.items()}

def get_object(name, position):
    if name == "burger_top":
        return BurgerTop(position)
    elif name == "burger_bottom":
        return BurgerBottom(position)
    elif name == "hamburger_patty":
        return HamburgerPatty(position)
    elif name == "lettuce":
        return Lettuce(position)
    elif name == "pickle_slice":
        return PickleSlice(position)
    elif name == "bread_slice" or name == "black_bread_slice" or name == "brown_bread_slice":
        return BreadSlice(position)
    elif name == "whole_cheese":
        return CheeseBlock(position)
    elif name == "whole_tomato":
        return WholeTomato(position)
    elif name == "whole_onion":
        return WholeOnion(position)
    elif name == "tomato_slice":
        return TomatoSlice(position)
    elif name == "onion_slice":
        return OnionSlice(position)
    elif name == "cheese_slice":
        return CheeseSlice(position)
    else:
        raise Exception(f"Invalid food name: {name}")

class Kitchen:
    """
    A class extended from `tdw.proc_gen.kitchen.py` to allow for more control over the kitchen generation process.
    """

    def create(self, ctrl: MacoController, seed: int = 42, is_test: bool = False):
        """
        Create a kitchen scene to ctrl. Return a list of objects that were added to the scene.
        """

        # Add the objects.
        self.pos_x = 0.5
        self.pos_z = -0.1
        table = CabinetTable({"x": self.pos_x, "y": 0, "z": self.pos_z}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8})
        size = table.original_size[0]
        height = table.original_size[1]
        # self.dist = size + 0.8

        self.tables = [
            table,
            CabinetTable({"x": self.pos_x - size * 3, "y": 0, "z": self.pos_z}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
            CabinetTable({"x": self.pos_x - size * 2, "y": 0, "z": self.pos_z}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
            CabinetTable({"x": self.pos_x - size, "y": 0, "z": self.pos_z}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
            CabinetTable({"x": self.pos_x, "y": 0, "z": self.pos_z + size}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
            CabinetTable({"x": self.pos_x, "y": 0, "z": self.pos_z + size * 2}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
            CabinetTable({"x": self.pos_x, "y": 0, "z": self.pos_z + size * 3}, rotation={"x": 0, "y": 270, "z": 0}, scale = {"x": 0.8, "y": 0.9, "z": 0.8}),
        ]

        ctrl.object_manager.add_objects(self.tables)

        self.rng = np.random.RandomState(seed)
        num_clip = 4
        num = [None, None]
        while True:
            num[0] = self.rng.randint(1, 4)
            num[1] = self.rng.randint(1, 4)
            if num[0] + num[1] == num_clip:
                break

        clip_food_candidates = [
            "hamburger_patty",
            "lettuce",
            "pickle_slice",
            "cheese_slice",
            "tomato_slice",
            "onion_slice",
            "cheese_slice",
            "tomato_slice",
            "onion_slice",
        ]
        self.added = [False for _ in range(len(clip_food_candidates))]
        clip_lst = [[], []]
        for i in range(2):
            for j in range(num[i]):
                idx = self.rng.choice([k for k in range(len(clip_food_candidates)) if not self.added[k]])
                clip_lst[i].append(clip_food_candidates[idx])
                for k in range(len(clip_food_candidates)):
                    if clip_food_candidates[k] == clip_food_candidates[idx]:
                        self.added[k] = True

        if self.rng.choice([True, False]):
            self.recipe = [["black_bread_slice"] + clip_lst[0] + ["brown_bread_slice"], ["burger_bottom"] + clip_lst[1] + ["burger_top"]]
        else:
            self.recipe = [["burger_bottom"] + clip_lst[0] + ["burger_top"], ["black_bread_slice"] + clip_lst[1] + ["brown_bread_slice"]]

        bins = [
            {"x": self.pos_x - 2 * size - 0.1, "y": height + 0.05, "z": self.pos_z + 0.1},
            {"x": self.pos_x - size + 0.1, "y": height + 0.05, "z": self.pos_z + 0.1},
            {"x": self.pos_x - 2 * size - 0.1, "y": height + 0.05, "z": self.pos_z - 0.1},
            {"x": self.pos_x - size + 0.1, "y": height + 0.05, "z": self.pos_z - 0.1},
            {"x": self.pos_x - 0.1, "y": height + 0.05, "z": self.pos_z + size - 0.1},
            {"x": self.pos_x - 0.1, "y": height + 0.05, "z": self.pos_z + 2 * size + 0.1},
            {"x": self.pos_x + 0.1, "y": height + 0.05, "z": self.pos_z + size - 0.1},
            {"x": self.pos_x + 0.1, "y": height + 0.05, "z": self.pos_z + 2 * size + 0.1},
        ]
        cutting_board_pos = {"x": self.pos_x, "y": height + 0.2, "z": self.pos_z}
        plates = [
            {"x": self.pos_x - 3 * size, "y": height + 0.05, "z": self.pos_z},
            {"x": self.pos_x, "y": height + 0.05, "z": self.pos_z + 3 * size}
        ]

        self.bins = copy.copy(bins)
        self.rng.shuffle(bins)
        self.food = []
        if is_test:
            proc = [0, 0]
        else:
            proc = [self.rng.randint(0, len(self.recipe[0]) + 1), self.rng.randint(0, len(self.recipe[1]) + 1)]

        for i in range(2):
            for j in range(proc[i]):
                pos = copy.copy(plates[i])
                pos["y"] += j * 0.1
                name = self.recipe[i][j]
                food = get_object(name, pos)
                if name == "black_bread_slice" or name == "brown_bread_slice":
                    food.name = name
                self.food.append(food)

            for j in range(proc[i], len(self.recipe[i])):
                pos = bins.pop()
                pos["x"] += self.rng.uniform(-0.02, 0.02)
                pos["z"] += self.rng.uniform(-0.02, 0.02)
                name = self.recipe[i][j]
                if name in SLICE_TO_WHOLE.keys():
                    name = self.rng.choice([name, SLICE_TO_WHOLE[name]])

                food = get_object(name, pos)
                if name == "black_bread_slice" or name == "brown_bread_slice":
                    food.name = name
                self.food.append(food)

        self.cutting_board = CuttingBoard({"x": self.pos_x, "y": height, "z": self.pos_z}, {"x": 0, "y": 90, "z": 0})
        self.knife = Knife({"x": self.pos_x, "y": height, "z": self.pos_z + size * 0.4}, {"x": 0, "y": 270, "z": 0})

        self.plates = [Plate({"x": self.pos_x - 3 * size, "y": height, "z": self.pos_z}),
                       Plate({"x": self.pos_x, "y": height, "z": self.pos_z + 3 * size})]

        ctrl.object_manager.add_objects(self.food)
        ctrl.object_manager.add_object(self.cutting_board)
        ctrl.object_manager.add_object(self.knife)
        ctrl.object_manager.add_objects(self.plates)
        ctrl.communicate([
            ctrl.get_add_scene("mm_kitchen_2a"),
			ctrl.get_add_material("rubber_floor_speckles"),
            {"$type": "set_floorplan_roof", "show": False},
            {"$type": "set_floor_material", "name": "rubber_floor_speckles"},
            # {"$type": "set_post_process", "value": False},
            {"$type": "step_physics", "frames": 50}
        ])

        ctrl.communicate([])
        
        material_name = "parquet_european_ash_grey"
        commands = [ctrl.get_add_material(material_name=material_name)]
        for tab in self.tables:
            commands.extend(TDWUtils.set_visual_material(c=ctrl, substructure=ModelLibrarian().get_record(name=tab.name).substructure, object_id=tab.id, material=material_name))
        ctrl.communicate(commands)

        # proc = ProcGenKitchen()
        # proc.create(scene="mm_kitchen_2a", rng=seed)
        # cmds = proc.commands
        # filtered_cmds = []
        # allowed_objects = []
        # banned = [-1e100, 0, -1e100, 1e100] # x_min, x_max, z_min, z_max
        # for cmd in cmds:
        #     if cmd["$type"] == "add_object":
        #         position = cmd["position"]
        #         position = [position["x"], position["y"], position["z"]]
        #         if position[0] >= banned[0] and position[0] <= banned[1] and position[2] >= banned[2] and position[2] <= banned[3]:
        #             continue
        #         if position[1] > 0.2:
        #             continue
        #         allowed_objects.append(cmd["id"])
        #         filtered_cmds.append(cmd)
        #     elif "id" in cmd:
        #         if cmd["id"] in allowed_objects:
        #             filtered_cmds.append(cmd)
        # ctrl.communicate(filtered_cmds)
        # cmds = []
        # for i in allowed_objects:
        #     cmds.append([{"$type": "set_kinematic_state", "id": i, "is_kinematic": True}])
        # ctrl.communicate(cmds)

        commands = [{"$type": "set_color", "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}, "id": self.knife.id}]
                    # {"$type": "set_color", "color": {"r": 1.0, "g": 1.0, "b": 0.0, "a": 1.0}, "id": self.plates[0].id},
                    # {"$type": "set_color", "color": {"r": 0.0, "g": 0.0, "b": 1.0, "a": 1.0}, "id": self.plates[1].id}]
        for food in self.food:
            if food.name == "black_bread_slice":
                commands.append({"$type": "set_color", "color": {"r": 0.4, "g": 0.4, "b": 0.4, "a": 1.0}, "id": food.id})
        
        ctrl.communicate(commands)