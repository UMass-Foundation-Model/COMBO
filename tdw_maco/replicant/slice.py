from enum import Enum
from typing import List
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.turn_to import TurnTo
from tdw.replicant.actions.grasp import Grasp

from replicant.reach_for import ReachFor
from replicant.state import ChallengeState
from replicant.reset_arms import ResetArms
from replicant.multi_action import MultiAction
from components.objects import Object
from components.object_manager import AdvancedObjectManager

# knife type: vk0007_steak_knife
"""
assumptions:
1. the agent is already holding the knife with arm
2. the target is already in position
3. the agent is standing near the target
4. target == sliced_target.id (Important for compatibility!!!)

when the action finishes, the agent will still be holding the knife with resetted arms

when num_slices is 3, the agent will move up, down, up, down, up, down, up
"""

class _SliceState(Enum):
    turning_to = 0
    slicing = 1
    resetting = 2

class Slice(MultiAction):
    def __init__(self, arm: Arm, target: int, sliced_target: Object, kinematic_objects: List[int], object_manager: AdvancedObjectManager, state: ChallengeState, num_slices: int = 5):
        super().__init__(state=state)
        self._arm: Arm = arm
        self._target: int = target
        self.kinematic_objects = kinematic_objects
        self._current_slice = 0
        self._num_slices = num_slices
        self._sliced_target = sliced_target
        self._slice_state: _SliceState = _SliceState.turning_to
        self._end_status: ActionStatus = ActionStatus.success
        self._object_manager = object_manager
    
    def _can_slice(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> bool:
        # Is the agent holding the knife?
        if self._arm not in dynamic.held_objects:
            return False
        # Is the object too far away?
        object_position = self._get_object_position(object_id=self._target, resp=resp)
        agent_position =  dynamic.transform.position
        object_position[1] = agent_position[1]
        # print(object_position, agent_position, np.linalg.norm(agent_position - object_position))
        if np.linalg.norm(agent_position - object_position) > 1.5:
            return False
        else:
            return True
    
    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic, image_frequency: ImageFrequency) -> List[dict]:
        if not self._can_slice(resp=resp, static=static, dynamic=dynamic):
            self.status = ActionStatus.already_holding
            return super().get_initialization_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=image_frequency)
        else:
            self._image_frequency = image_frequency
            self._sub_action = TurnTo(target=self._target)
            ret = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=image_frequency)
            target_position = self._get_object_bounds(object_id=self._target, resp=resp)["bottom"]
            offset = (target_position - dynamic.transform.position) * np.array([1, 0, 1])
            offset /= np.linalg.norm(offset)
            self.slice_up_position = target_position + np.array([0, 0.25, 0])
            self.slice_down_position = target_position + np.array([0, 0.05, 0])
            self.slice_up_position -= offset * 0.1
            self.slice_down_position -= offset * 0.1
            ret.append({"$type": "set_kinematic_state", "id": self._target, "is_kinematic": True, "use_gravity": False})
            return ret
    
    def _reset(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic, success=False) -> List[dict]:
        self._sub_action = ResetArms(state=self._state)
        self._slice_state = _SliceState.resetting
        ret = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=self._image_frequency)
        if success:
            ret.append({"$type": "destroy_object", "id": self._target})
            self._object_manager.add_object(self._sliced_target)
        return ret
    
    def _slice_up(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        self._slice_state = _SliceState.slicing
        self._current_slice += 1
        self._sub_action = ReachFor(target=self.slice_up_position, arm=self._arm, dynamic=dynamic, absolute=True)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=self._image_frequency)
    
    def _slice_down(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        self._slice_state = _SliceState.slicing
        self._current_slice += 1
        self._sub_action = ReachFor(target=self.slice_down_position, arm=self._arm, dynamic=dynamic, absolute=True)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic, image_frequency=self._image_frequency)
    
    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        if self._slice_state != _SliceState.resetting and not self._can_slice(resp=resp, static=static, dynamic=dynamic):
            self._end_status = ActionStatus.not_holding
            return self._reset(resp=resp, static=static, dynamic=dynamic)
        
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success:
            if self._slice_state == _SliceState.turning_to:
                return self._slice_up(resp=resp, static=static, dynamic=dynamic)
            elif self._slice_state == _SliceState.slicing:
                if self._current_slice >= self._num_slices * 2 + 1:
                    return self._reset(resp=resp, static=static, dynamic=dynamic, success=True)
                elif self._current_slice % 2 == 0:
                    return self._slice_up(resp=resp, static=static, dynamic=dynamic)
                else:
                    return self._slice_down(resp=resp, static=static, dynamic=dynamic)
            elif self._slice_state == _SliceState.resetting:
                self.status = self._end_status
                return []
        else:
            self._end_status = self._sub_action.status
            return self._reset(resp=resp, static=static, dynamic=dynamic)