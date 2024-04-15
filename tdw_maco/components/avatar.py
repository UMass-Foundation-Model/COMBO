from typing import List, Union
from tdw.add_ons.add_on import AddOn

class Avatar(AddOn):
    def __init__(self, name: str, position: dict, look_at: dict, render_order: int, rotation: dict = {"x": 0, "y": 0, "z": 0}):
        super().__init__()
        self.name = name
        self.position = position
        self.look_at = look_at
        self.render_order = render_order
        self.rotation = rotation
    
    def get_initialization_commands(self):
        return [{
            "$type": "create_avatar",
            "type": "A_Img_Caps_Kinematic",
            "id": self.name
        },{
            "$type": "set_pass_masks",
            "pass_masks": [ "_img", "_depth"],
            "avatar_id": self.name 
        }, {
            "$type": "set_render_order",
            "render_order": self.render_order,
            "avatar_id": self.name
        }, {
            "$type": "teleport_avatar_to",
            "position": self.position,
            "avatar_id": self.name
        }, {
            "$type": "look_at_position",
            "position": self.look_at,
            "avatar_id": self.name
        },{
            "$type": "rotate_avatar_to_euler_angles",
            "euler_angles": self.rotation,
            "avatar_id": self.name
        },]
    
    def on_send(self, resp: List[bytes]) -> None:
        pass

    def before_send(self, commands: List[dict]) -> None:
        pass

class TopDownAvatar(Avatar):
    def __init__(self, position: dict = {"x": 0, "y": 6, "z": 0}, render_order: int = 120):
        look_at = {"x": position["x"], "y": 0, "z": position["z"]}
        super().__init__("top_down", position, look_at, render_order)

class TeaserAvatar(Avatar):
    def __init__(self, position: dict = {"x": -0.7, "y": 3.5, "z": -3}, render_order: int = 110):
        look_at = {"x": position["x"], "y": 0, "z": 0}
        super().__init__("teaser", position, look_at, render_order)

class TopDownAvatarRotate(Avatar):
    def __init__(self, position: dict = {"x": 0, "y": 6, "z": 0}, render_order: int = 120, rotation = {"x": 0, "y": 90, "z": 0}):
        look_at = {"x": position["x"], "y": 0, "z": position["z"]}
        super().__init__("top_down", position, look_at, render_order, rotation)

class CookTopDownAvatar(Avatar):
    def __init__(self, position: dict = {"x": 0, "y": 4.2, "z": 0.2}, render_order: int = 120, rotation = {"x": 0, "y": 270, "z": 0}):
        look_at = {"x": position["x"], "y": 0, "z": position["z"]}
        super().__init__("top_down", position, look_at, render_order, rotation)
