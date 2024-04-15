from typing import List
import numpy as np
from tdw.add_ons.composite_object_manager import CompositeObjectManager
from collections import defaultdict

from envs.maco_controller import MacoController
from components.objects import *
import json
import glob
    
angle_4 = [180, 90, 0, -90]
angle_3 = [180, 90, 0]
angle_2 = [180, 90]

class CraftRoom:
    def create(self, ctrl: MacoController, number_of_agents, seed = 42, scale: dict = {"x": 0.8, "y": 0.8, "z": 0.8}, is_test = False):
        ctrl.communicate([
            ctrl.get_add_scene("mm_craftroom_2a"),
            ctrl.get_add_material("rubber_floor_speckles"),
            {"$type": "set_floorplan_roof", "show": False},
            {"$type": "set_floor_material", "name": "rubber_floor_speckles"},
            {"$type": "set_post_process", "value": False},
            {"$type": "step_physics", "frames": 50},

        ])
        self.table = SquareTable(scale={"x": 2 * 3.2 / 4.2, "y": 1,
                                            "z": 2 * 3.2 / 4.2})

        ctrl.object_manager.add_object(self.table)
        # need two communicates to return static data
        ctrl.communicate([])
        ctrl.communicate([])
        self.size = self.table.static.size[0]
        self.height = self.table.static.size[1]
        # print(self.size)

        self.on_table = self.size / 2 - 0.25
        self.gaps = [-self.size / 2 + 0.25 + (self.size - 0.5) / 4 * i for i in range(5) if i != 2]
        self.gaps[1] -= 0.05
        self.gaps[2] += 0.05
        if number_of_agents == 4:
            self.bins = [{"x": gap, "y": self.height, "z": self.on_table} for gap in self.gaps] + \
               [{"x": gap, "y": self.height, "z": -self.on_table} for gap in self.gaps] + \
               [{"x": self.on_table, "y": self.height, "z": gap} for gap in self.gaps[1:-1]] + \
               [{"x": -self.on_table, "y": self.height, "z": gap} for gap in self.gaps[1:-1]]  # 12 bins for scattering pieces
        elif number_of_agents == 3:
            self.bins = [{"x": gap, "y": self.height, "z": self.on_table} for gap in self.gaps] + \
               [{"x": gap, "y": self.height, "z": -self.on_table} for gap in self.gaps] + \
               [{"x": -self.on_table, "y": self.height, "z": gap} for gap in self.gaps[1:-1]] 
        elif number_of_agents == 2:
            self.bins = [{"x": gap, "y": self.height, "z": self.on_table} for gap in self.gaps] + \
               [{"x": -self.on_table, "y": self.height, "z": gap} for gap in self.gaps[:-1]] 

        rng = np.random.RandomState(seed)
        rng.shuffle(self.bins)

        # self.bins = [
        #         {"x": -0.8, "y": self.height, "z": -0.8},
        #         {"x": -0.8, "y": self.height, "z": 0.8},
        #         {"x": 0.8, "y": self.height, "z": -0.8},
        #         {"x": 1.2, "y": self.height, "z": 0.8},
        #         {"x": -0.4, "y": self.height, "z": -0.8},
        #         {"x": -0.4, "y": self.height, "z": 0.8},
        #         {"x": -0.8, "y": self.height, "z": -0.5},
        #         {"x": 0.4, "y": self.height, "z": 0.8},
        #         {"x": 0.4, "y": self.height, "z": -0.8},
        #         {"x": 0.8, "y": self.height, "z": 0.4},
        #         {"x": -0.8, "y": self.height, "z": 0.4},
        #         {"x": 0.8, "y": self.height, "z": -0.4},
        #     ]
            
        if number_of_agents == 4:
            permutation = [1, 2, 3, 4]
            rng.shuffle(permutation)
            self.puzzles: List[Object] = [Puzzle(permutation[0], {"x": 0, "y": self.height, "z": self.on_table}, scale = scale),
                                        Puzzle(permutation[1], {"x": -self.on_table, "y": self.height, "z": 0}, scale = scale),
                                        Puzzle(permutation[2], {"x": 0, "y": self.height, "z": -self.on_table}, scale = scale),
                                        Puzzle(permutation[3], {"x": self.on_table, "y": self.height, "z": 0}, scale = scale)]
            
        elif number_of_agents == 3:
            permutation = [1, 2, 3, 4]
            rng.shuffle(permutation)
            self.puzzles: List[Object] = [Puzzle(permutation[0], {"x": 0, "y": self.height, "z": self.on_table}, scale = scale),
                                        Puzzle(permutation[1], {"x": -self.on_table, "y": self.height, "z": 0}, scale = scale),
                                        Puzzle(permutation[2], {"x": 0, "y": self.height, "z": -self.on_table}, scale = scale)]

        elif number_of_agents == 2:
            permutation = [1, 2, 3, 4]
            rng.shuffle(permutation)
            self.puzzles: List[Object] = [Puzzle(permutation[0], {"x": 0, "y": self.height, "z": self.on_table}, scale = scale),
                                        Puzzle(permutation[1], {"x": -self.on_table, "y": self.height, "z": 0}, scale = scale)]

        ctrl.object_manager.add_objects(self.puzzles)
        self.composite_object_manager = CompositeObjectManager()
        ctrl.add_ons.append(self.composite_object_manager)
        ctrl.communicate([])
        self.puzzle_id2piece_id = defaultdict(list)
        for object_id in self.composite_object_manager.static:
            composite_object_static = self.composite_object_manager.static[object_id]
            for sub_object_id in composite_object_static.non_machines:
                self.puzzle_id2piece_id[object_id].append(
                    composite_object_static.non_machines[sub_object_id].sub_object_id)
                
        ctrl.object_manager.reset()
        ctrl.communicate([])
        self.name2id = dict()
        for i, puzzle in enumerate(self.puzzles):
            for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
                piece_id = piece
                self.name2id[ctrl.object_manager.objects_static[piece_id].name] = piece_id

        commands = []
        for i, puzzle in enumerate(self.puzzles):
            commands.append({"$type": "set_kinematic_state",
                        "id": puzzle.id,
                        "is_kinematic": True})
        
        ctrl.communicate(commands)

        puzzle_size = self.puzzles[0].static.size

        self.relative_position_dict = dict()
        self.check_for_pos_dict = dict()
        self.annotation_dict = dict()
        self.semantic_name_dict = dict()
        self.model_name_to_semantic_name_dict = dict()
        json_files = glob.glob('./local_asset/shapes_puzzle_*.json')
        for file in json_files:
            cur_dict = json.load(open(file, "r"))
            for d in cur_dict.values():
                assert "model_name" in d and "position" in d, f"json file information error!"
                name = d['model_name'].lower()
                
                if name not in self.name2id:
                    continue
                
                self.relative_position_dict[self.name2id[name]] = np.array([d["position"]["x"] / d["scale"]["x"] * puzzle_size[0] * 2, d["position"]["y"] / d["scale"]["y"] * puzzle_size[1] + 0.01, d["position"]["z"] / d["scale"]["z"] * puzzle_size[2] * 2])
                self.annotation_dict[self.name2id[name]] = d["annotation"]
                self.semantic_name_dict[self.name2id[name]] = d["semantic_name"]
                self.model_name_to_semantic_name_dict[name] = d["semantic_name"]

        commands = []
        if number_of_agents == 4:
            for i, puzzle in enumerate(self.puzzles):
                commands.append({"$type": "rotate_object_by",
                                            "id": puzzle.id,
                                            "axis": "yaw",
                                            "angle": angle_4[i]})
        elif number_of_agents == 3:
            for i, puzzle in enumerate(self.puzzles):
                commands.append({"$type": "rotate_object_by",
                                            "id": puzzle.id,
                                            "axis": "yaw",
                                            "angle": angle_3[i]})
        elif number_of_agents == 2:
            for i, puzzle in enumerate(self.puzzles):
                commands.append({"$type": "rotate_object_by",
                                            "id": puzzle.id,
                                            "axis": "yaw",
                                            "angle": angle_2[i]})
            
        for i, puzzle in enumerate(self.puzzles):
            for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
                piece_id = piece
                commands.append({"$type": "set_kinematic_state",
                                "id": piece_id,
                                "is_kinematic": True})
                
        piece_lst = []
        for i, puzzle in enumerate(self.puzzles):
            for j, piece in enumerate(self.puzzle_id2piece_id[puzzle.id]):
                piece_id = piece
                if len(self.annotation_dict[piece_id]) > 0:
                    piece_lst.append(piece_id)

        rng.shuffle(piece_lst)

        n_piece = len(piece_lst) # number of pieces that can get out of the puzzle

        if is_test or force_initial or rng.randint(1, 4) == 1:
            if number_of_agents == 2:
                num_out = min(4, n_piece)
            elif number_of_agents == 3:
                num_out = min(6, n_piece)
            else:
                num_out = min(8, n_piece)
        else:
            if number_of_agents == 2:
                num_out = rng.randint(1, min(4, n_piece) + 1)
            elif number_of_agents == 3:
                num_out = rng.randint(1, min(6, n_piece) + 1)
            else:
                num_out = rng.randint(1, min(8, n_piece) + 1)

        for i in range(num_out):
            pos = self.bins.pop()
            pos["x"] += rng.uniform(low=-0.02, high=0.02)
            pos["z"] += rng.uniform(low=-0.02, high=0.02)  # small deviation
            commands.append({"$type": "teleport_object", "position": pos, "id": piece_lst[i]})
                
        ctrl.communicate(commands)