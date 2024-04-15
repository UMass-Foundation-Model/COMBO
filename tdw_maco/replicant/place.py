from typing import Union, List
from enum import Enum
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
# from tdw.replicant.actions.reach_for import ReachFor
from tdw.replicant.actions.drop import Drop
from tdw.replicant.actions.turn_to import TurnTo
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.image_frequency import ImageFrequency
import copy

from replicant.multi_action import MultiAction
from replicant.state import ChallengeState
from replicant.reset_arms import ResetArms
from replicant.reach_for import ReachFor
from tdw.tdw_utils import TDWUtils

class _PlaceState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """
    turning = 0
    reach_for = 1
    spinning = 2
    drop = 3
    reset_arms = 4
    end = 5

    


class Place(MultiAction):
    def __init__(self, arm: Arm, position: Union[np.ndarray, dict], state: ChallengeState, object_id: int, set_kinematic_state: bool = False, skip_drop: bool = False, object_rotate_angle: float = None):
        super().__init__(state)
        self._arm = arm
        self.teleport_pos = copy.deepcopy(position)

        if isinstance(position, dict):
            position["y"] += 0.2
        else:
            position[1] += 0.2
        self._position = np.array(position)
        self._sub_action = None
        self._end_status = ActionStatus.success
        self._place_state = _PlaceState.turning
        self.object_id = object_id
        self._set_kinematic_state = set_kinematic_state
        self._skip_drop = skip_drop
        self._object_rotate_angle = object_rotate_angle


    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic, image_frequency: ImageFrequency) -> list:
        # Is a Replicant already holding this object?
        if not self._can_drop(dynamic=dynamic):
            self.status = ActionStatus.not_holding
            return []
        # Turn to face the object.
        else:
            self._image_frequency = image_frequency
            self._sub_action = TurnTo(target=self._position)
            self._place_state = _PlaceState.turning
            return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                image_frequency=image_frequency)
    
    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """

        # Continue an ongoing sub-action.
        # print(self._sub_action, self._sub_action.status)
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success or self._sub_action.status == ActionStatus.collision or self._sub_action.status == ActionStatus.still_dropping:
            if self._place_state == _PlaceState.turning:
                # self._sub_action = ReachFor(targets=[self._position],
                #                         absolute=True,
                #                         offhand_follows=False,
                #                         arrived_at=0,
                #                         max_distance=1.5,
                #                         arms=[self._arm],
                #                         dynamic=dynamic,
                #                         collision_detection=CollisionDetection(objects=False, held=False),
                #                         previous=None,
                #                         duration=0.5,
                #                         scale_duration=True,
                #                         from_held=False,
                #                         held_point="center")
                # return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                #                                                     image_frequency=self._image_frequency)
                self._place_state = _PlaceState.reach_for
                self._sub_action = ReachFor(target=self._position,
                                                      arm=self._arm,
                                                      dynamic=dynamic,
                                                      absolute=True)
                return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
            elif self._place_state == _PlaceState.reach_for:
                self._place_state = _PlaceState.spinning
                if self._object_rotate_angle is not None:
                    return [{"$type": "rotate_object_by", "id": self.object_id, "angle": self._object_rotate_angle, "axis": "yaw"}]
                else:
                    return []
            elif self._place_state == _PlaceState.spinning:
                if self._skip_drop:
                    self._place_state = _PlaceState.reset_arms
                    self._sub_action = Drop(arm=self._arm, dynamic=dynamic, max_num_frames=25, offset=0)
                    cmds = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                        image_frequency=self._image_frequency)
                    
                    cmds.append({
                        "$type": "teleport_object",
                        "id": self.object_id,
                        "position": TDWUtils.array_to_vector3(self.teleport_pos)
                    })

                    self._sub_action = ResetArms(state=self._state)
                    cmds.extend(self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                        image_frequency=self._image_frequency))
                    
                else:
                    self._place_state = _PlaceState.drop
                    self._sub_action = Drop(arm=self._arm, dynamic=dynamic, max_num_frames=25, offset=0)
                    cmds = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                        image_frequency=self._image_frequency)
                
                return cmds
                
            elif self._place_state == _PlaceState.drop:
                self._place_state = _PlaceState.reset_arms
                self._sub_action = ResetArms(state=self._state)
                cmds = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
                
                cmds.append({
                    "$type": "teleport_object",
                    "id": self.object_id,
                    "position": TDWUtils.array_to_vector3(self.teleport_pos)
                })
                
                return cmds
            elif self._place_state == _PlaceState.reset_arms:
                self._place_state = _PlaceState.end
                if self._set_kinematic_state:
                    return [{"$type": "set_kinematic_state", "id": self.object_id, "is_kinematic": True, "use_gravity": False}]
                else:
                    return []
            elif self._place_state == _PlaceState.end:
                self.status = self._end_status
                return []
        # We failed.
        else:
            # Remember the fail status.
            self._end_status = self._sub_action.status
            return self._reset_arms(resp=resp, static=static, dynamic=dynamic)
    
    def _can_drop(self, dynamic: ReplicantDynamic) -> bool:
        """
        :param dynamic: The `ReplicantDynamic` data that changes per `communicate()` call.

        :return: True if the Replicant is holding the object.
        """

        return self._arm in dynamic.held_objects
    
    def _reset_arms(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reset the arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ResetArms(state=self._state)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)