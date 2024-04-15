from typing import Union, List
from enum import Enum
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
# from tdw.replicant.actions.reach_for import ReachFor

from tdw.replicant.actions.look_at import LookAt
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.image_frequency import ImageFrequency
import copy
from tdw.type_aliases import TARGET

from replicant.multi_action import MultiAction
from replicant.state import ChallengeState
from replicant.reset_arms import ResetArms
from replicant.reach_for import ReachFor
from tdw.tdw_utils import TDWUtils

class _LookatState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """
    looking = 0
    step = 1
    end = 2

    


class LookTo(MultiAction):
    def __init__(self, target: TARGET, duration: float, scale_duration: bool, state: ChallengeState):
        super().__init__(state)
        self._target = target
        self._duration = duration
        self._scale_duration = scale_duration
        self._end_status = ActionStatus.success


    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic, image_frequency: ImageFrequency) -> list:
        self._sub_action = LookAt(target=self._target, duration=self._duration, scale_duration=self._scale_duration)
        self._look_at_state = _LookatState.looking
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
        elif self._look_at_state == _LookatState.looking:
            self.status = self._end_status
            return []