from pathlib import Path
from typing import Union
from tdw.controller import Controller
from tdw.add_ons.object_manager import ObjectManager
import numpy as np
from tdw.controller import Controller

class Object:
    def __init__(self, name: str, position: dict, rotation: dict, scale: dict = None, custom=False, library="models_core.json", category: Union[str, None] = None, is_kinematic: bool = False, mass: int = 1, id = None):
        self.initialized = True
        self.id = Controller.get_unique_id() if id is None else id
        self.name = name
        self.position = position
        self.rotation = rotation
        self.scale = {x: 1 for x in ["x", "y", "z"]}
        self.commands = []

        if not custom:
            self.commands.extend(Controller.get_add_physics_object(model_name=name, object_id=self.id, position=self.position, rotation=self.rotation, library=library, kinematic=is_kinematic))
            self.record = Controller.MODEL_LIBRARIANS[library].get_record(name)

        else:
            
            self.record = None
            assert category is not None, "Category must be specified for custom objects."

            asset_bundle_path = Path("local_asset/Linux").joinpath(self.name)
            asset_bundle_url = "file:///" + str(asset_bundle_path.resolve()).replace("\\", "/")
            self.commands.append({
                "$type": "add_object",
                "name": self.name,
                "url": asset_bundle_url,
                "scale_factor": 1,
                "position": self.position,
                "rotation": self.rotation,
                "category": category,
                "mass": mass,
                "dynamic_friction": 0.95,
                "static_friction": 0.95,
                "bounciness": 0.01,
                "default_physics_values": False, 
                "is_kinematic": is_kinematic,
                "abc": True,
                "id": self.id})

        if scale is not None:
            self._scale_object(scale)
    
    def _scale_object(self, scale: dict):
        """
        should only be called once during initialization
        """
        self.scale = {x: scale[x] * self.scale[x] for x in scale.keys()} if self.scale is not None else scale
        self.commands.append({"$type": "scale_object_and_mass", "scale_factor": scale, "id": self.id})
    
    def _rotate_object(self, rotation: dict):
        self.commands.append({"$type": "rotate_object_to_euler_angles", "euler_angles": rotation, "id": self.id})
    
    @property
    def original_size(self):
        """
        Returns the original size of the object in TDW units(include scale but ignore rotation).
        """
        size = np.abs(np.array([self.record.bounds["right"]["x"] - self.record.bounds["left"]["x"],
                                self.record.bounds["top"]["y"] - self.record.bounds["bottom"]["y"],
                                self.record.bounds["front"]["z"] - self.record.bounds["back"]["z"]]))
        if self.scale is not None:
            size *= np.array(list(self.scale.values()))
        return size
    
    def _set_manager(self, manager: ObjectManager):
        self.manager = manager
    
    @property
    def static(self):
        return self.manager.objects_static[self.id]
    
    @property
    def bound(self):
        return self.manager.bounds[self.id]
    
    @property
    def transform(self):
        return self.manager.transforms[self.id]

class SquareObject(Object):
    def __init__(self, name: str, position: dict, rotation: dict = None, scale: dict = None,  custom=False, library="models_core.json", category: Union[str, None] = None, is_kinematic: bool = False, mass: int = 1, id=None):
        super().__init__(name, position, rotation, scale, custom, library, category, is_kinematic, mass, id=id)
        x, y, z = self.original_size
        l = max(x, z)
        self._scale_object({"x": l / x, "y": 1, "z": l / z})
        # if rotation is not None:
        #     self._rotate_object(rotation)

class WoodBench(SquareObject):
    def __init__(self, position: dict ,rotation = None, scale: dict = {"x": 0.3, "y": 1, "z": 0.3}, id=None):
        super().__init__("bench", position, rotation, scale, id=id)

class SquareTable(SquareObject):
    def __init__(self, position: dict = {"x": 0, "y": 0, "z": 0}, rotation = None, scale: dict = None, id=None):
        super().__init__("b05_table_new", position, rotation, scale, custom=False, category="table", is_kinematic=True, mass = 1, id=id)

class CabinetTable(SquareObject):
    def __init__(self, position: dict = {"x": 0, "y": 0, "z": 0}, rotation = None, scale: dict = None, id=None):
        super().__init__("cabinet_24_two_drawer_wood_beach_honey", position, rotation, scale, custom=False, category="table", is_kinematic=True, mass = 1000) #, id=id cabinet_24_two_door_wood_oak_white_composite

class TriangleTable(Object):
    def __init__(self, position: dict = {"x": 0, "y": 0, "z": 0}, rotation={"x": 0, "y": 0, "z": 0}, scale: dict = None, id=None):
        super().__init__("triangle_table", position, rotation, None, custom=True, category="table", id=id)

class CabinetFloat(Object):
    def __init__(self, position: dict, table_size: float, rotation={"x": 0, "y": 0, "z": 0}, id=None):
        super().__init__("floating_counter_top_counter_top", position, rotation, None, custom=True, category="table", library="models_special.json", id=id)
        x, y, z = self.original_size
        self._scale_object({"x": table_size / x, "y": 1, "z": table_size / z})

class CuttingBoard(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.5, "y": 3, "z": 0.4}, id=None):
        super().__init__("wood_board", position, rotation, scale, id=id, is_kinematic=True, mass=1000)

class BreadSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.8, "y": 1.8, "z": 1.8}, id=None):
        super().__init__("bread_slice", position, rotation, scale, custom=True, category="bread_slice", id=id)

class CheeseSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.0, "y": 2.0, "z": 2.0}, id=None):
        super().__init__("cheese_slice", position, rotation, scale, custom=True, category="cheese_slice", id=id, is_kinematic=True)

class CheeseBlock(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.0, "y": 2.0, "z": 2.0}, id=None):
        super().__init__("whole_cheese", position, rotation, scale, custom=True, category="whole_cheese", id=id)

class HamSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.8, "y": 1.8, "z": 1.8}, id=None):
        super().__init__("ham_slice", position, rotation, scale, custom=True, category="ham_slice", id=id)

class TomatoSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 3, "y": 3, "z": 3}, id=None):
        super().__init__("tomato_slice", position, rotation, scale, custom=True, category="tomato_slice", id=id, is_kinematic=True)

class WholeTomato(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 3, "y": 3, "z": 3}, id=None):
        super().__init__("whole_tomato", position, rotation, scale, custom=True, category="whole_tomato", id=id)

class PickleSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 3.5, "y": 3.5, "z": 3.5}, id=None):
        super().__init__("pickle_slice", position, rotation, scale, custom=True, category="pickle_slice", id=id,)

class Lettuce(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.5, "y": 2.5, "z": 2.5}, id=None):
        super().__init__("lettuce", position, rotation, scale, custom=True, category="lettuce", id=id)

class BurgerTop(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.8, "y": 1.8, "z": 1.8}, id=None):
        super().__init__("burger_top", position, rotation, scale, custom=True, category="burger_top", id=id)

class BurgerBottom(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.8, "y": 1.8, "z": 1.8}, id=None):
        super().__init__("burger_bottom", position, rotation, scale, custom=True, category="burger_bottom", id=id)

class HamburgerPatty(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.2, "y": 2.2, "z": 2.2}, id=None):
        super().__init__("hamburger_patty", position, rotation, scale, custom=True, category="hamburger_patty", id=id)

class WholeOnion(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.5, "y": 2.5, "z": 2.5}, id=None):
        super().__init__("whole_onion", position, rotation, scale, custom=True, category="whole_onion", id=id)    

class OnionSlice(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.5, "y": 2.5, "z": 2.5}, id=None):
        super().__init__("onion_slice", position, rotation, scale, custom=True, category="onion_slice", id=id, is_kinematic=True)       

# class Puzzle_1(Object):
#     def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.2, "y": 1.2, "z": 1.2}):
#         super().__init__("shapes_puzzle_physics_v1_piecemapped", position, rotation, scale, custom=True, category="shape_puzzle", is_kinematic=True)

# class Puzzle_2(Object):
#     def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.2, "y": 1.2, "z": 1.2}):
#         super().__init__("shapes_puzzle_physics_v2_piecemapped", position, rotation, scale, custom=True, category="shape_puzzle", is_kinematic=True)

# class Puzzle_3(Object):
#     def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.2, "y": 1.2, "z": 1.2}):
#         super().__init__("shapes_puzzle_physics_v3_piecemapped", position, rotation, scale, custom=True, category="shape_puzzle", is_kinematic=True)

# class Puzzle_4(Object):
#     def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.2, "y": 1.2, "z": 1.2}):
#         super().__init__("shapes_puzzle_physics_v4_piecemapped", position, rotation, scale, custom=True, category="shape_puzzle", is_kinematic=True)

class Puzzle(Object):
    def __init__(self, idx: int, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 0.8, "y": 0.8, "z": 0.8}, id=None):
        super().__init__(f"shapes_puzzle_physics_v{idx}_piecemapped", position, rotation, scale, custom=True, category="shape_puzzle", is_kinematic=True, id=id)

class Knife(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 1.5, "y": 1.5, "z": 1.5}, id=None):
        super().__init__("vk0007_steak_knife", position, rotation, scale=scale, category="knife", id=id)

class Plate(Object):
    def __init__(self, position: dict, rotation: dict={"x": 0, "y": 0, "z": 0}, scale: dict = {"x": 2.5, "y": 2.5, "z": 2.5}, id=None):
        super().__init__("plate05", position, rotation, scale=scale, category="plate", id=id, is_kinematic=True)