from typing import List, Dict
from tdw.add_ons.object_manager import ObjectManager

from components.objects import Object

class AdvancedObjectManager(ObjectManager):
    def __init__(self, transforms: bool = True, rigidbodies: bool = False, bounds: bool = True):
        super().__init__(transforms, rigidbodies, bounds)
        self.objects_by_id: Dict[int, Object] = {}
        self.objects_by_name: Dict[str, Object] = {}
        self._new_objects = []
        self._last_new_objects = []
    
    def add_object(self, obj: Object):
        obj._set_manager(self)
        self.objects_by_id[obj.id] = obj
        self.objects_by_name[obj.name] = obj
        self._new_objects.append(obj.id)
        self.commands.extend(obj.commands)
    
    def add_objects(self, objs: List[Object]):
        for obj in objs:
            self.add_object(obj)
    
    def before_send(self, commands: List[dict]) -> None:
        if len(self._last_new_objects) > 0:
            self._cached_static_data = False
            commands.extend([{"$type": "send_segmentation_colors"},
                            {"$type": "send_categories"},
                            {"$type": "send_static_rigidbodies"},
                            {"$type": "send_rigidbodies", "frequency": self._send_rigidbodies},
                            {"$type": "send_bounds", "frequency": self._send_bounds},
                            {"$type": "send_transforms", "frequency": self._send_transforms}])
        self._last_new_objects = self._new_objects
        self._new_objects = []
