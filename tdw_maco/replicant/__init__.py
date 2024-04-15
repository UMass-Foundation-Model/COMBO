from typing import Dict, Union, List
import numpy as np
from tdw.add_ons.replicant import Replicant as ReplicantBase
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.arm import Arm
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.actions.do_nothing import DoNothing
from tdw.replicant.collision_detection import CollisionDetection
from tdw.output_data import StaticRigidbodies, OutputData

from replicant.pick_up import PickUp
from replicant.put_in import PutIn
from replicant.state import ChallengeState
from replicant.navigate_to import NavigateTo
from replicant.reset_arms import ResetArms
from replicant.place import Place
from replicant.look_at import LookTo
import copy
from replicant.slice import Slice
from components.objects import Object
from components.object_manager import AdvancedObjectManager

class Replicant(ReplicantBase):
    """
    A wrapper class for `Replicant` for the Transport Challenge.

    This class is a subclass of `Replicant`. It includes the entire `Replicant` API plus specialized Transport Challenge actions. Only the Transport Challenge actions are documented here. For the full Replicant documentation, [read this.](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/replicants/overview.md)

    ![](images/action_space.jpg)
    """

    def __init__(self, replicant_id: int, state: ChallengeState,
                 home_look_at: np.ndarray = np.array([0, 0, 0]),
                 turn_left: np.ndarray = np.array([0, 0, 0]),
                 position: Union[Dict[str, float], np.ndarray] = None,
                 rotation: Union[Dict[str, float], np.ndarray] = None,
                 image_frequency: ImageFrequency = ImageFrequency.once,
                 target_framerate: int = 250,
                 enable_collision_detection: bool = False,
                 name: str = "replicant_0",
                 ):
        """
        :param replicant_id: The ID of the Replicant.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        :param position: The position of the Replicant as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param rotation: The rotation of the Replicant in Euler angles (degrees) as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value that sets how often images are captured.
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        """

        super().__init__(replicant_id=replicant_id, position=position, rotation=rotation,
                         image_frequency=image_frequency, target_framerate=target_framerate, name=name)
        self._state: ChallengeState = state
        self.collision_detection.held = False
        self.collision_detection.previous_was_same = False
        self.home_position = position
        self.home_rotation = rotation
        self.home_look_at = home_look_at
        self.turn_left = turn_left
        self.left = self.home_look_at + self.turn_left
        self.right = self.home_look_at - self.turn_left

    def turn_by(self, angle: float) -> None:
        """
        Turn the Replicant by an angle.

        :param angle: The target angle in degrees. Positive value = clockwise turn.
        """

        super().turn_by(angle=angle)

    def turn_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Turn the Replicant to face a target object or position.

        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        """

        super().turn_to(target=target)

    def drop(self, arm: Arm, max_num_frames: int = 100, offset: float = 0.1) -> None:
        """
        Drop a held target object.

        The action ends when the object stops moving or the number of consecutive `communicate()` calls since dropping the object exceeds `self.max_num_frames`.

        When an object is dropped, it is made non-kinematic. Any objects contained by the object are parented to it and also made non-kinematic. For more information regarding containment in TDW, [read this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/semantic_states/containment.md).

        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) holding the object.
        :param max_num_frames: Wait this number of `communicate()` calls maximum for the object to stop moving before ending the action.
        :param offset: Prior to being dropped, set the object's positional offset. This can be a float (a distance along the object's forward directional vector). Or it can be a dictionary or numpy array (a world space position).
        """

        super().drop(arm=arm, max_num_frames=max_num_frames, offset=offset)

    def move_forward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance forward. This calls `self.move_by(distance)`.

        :param distance: The distance.
        """

        super().move_by(distance=abs(distance), reset_arms=False)

    def move_backward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance backward. This calls `self.move_by(-distance)`.

        :param distance: The distance.
        """

        super().move_by(distance=-abs(distance), reset_arms=False)

    def move_to_object(self, target: int) -> None:
        """
        Move to an object. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        self.move_to(target=target, reset_arms=False,
                     arrived_at=0.7)  # if target in self._state.target_object_ids else 0.1)

    def move_to_position(self, target: np.ndarray) -> None:
        """
        Move to an position. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        assert isinstance(target, np.ndarray) and target.shape == (3,), f"target must be a 3D numpy array. Got {target}"
        target[1] = 0
        self.move_to(target=target, reset_arms=False, arrived_at=0.7)

    def pick_up(self, target: int, arm = None, offset: int = 0, reach_pos: np.array = None, set_kinematic_state: bool = False, object_rotate_angle = None) -> None:
        """
        Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

        The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

        See: [`PickUp`](pick_up.md)

		:param target: The object ID.
		"""
        
        if arm is None:
            arm=Arm.left if Arm.right in self.dynamic.held_objects else Arm.right

        self.action = PickUp(arm=arm, target=target, kinematic_objects=self._kinematic_objects, state=self._state, offset = offset, reach_pos = reach_pos, set_kinematic_state = set_kinematic_state, object_rotate_angle = object_rotate_angle)
    
    def slice(self, target: int, sliced_target: Object, object_manager: AdvancedObjectManager, arm = None, num_times=5) -> None:
        # self.collision_detection = CollisionDetection(exclude_objects=[target])
        if arm is None:
            arm=Arm.left if Arm.left in self.dynamic.held_objects else Arm.right
        self.action = Slice(arm=arm, target=target, sliced_target=sliced_target, kinematic_objects=self._kinematic_objects, object_manager=object_manager, state=self._state, num_slices=num_times)

    def put_in(self) -> None:
        """
        Put an object in a container.

        The Replicant must already be holding the container in one hand and the object in the other hand.

        See: [`PutIn`](put_in.md)
        """

        self.action = PutIn(dynamic=self.dynamic, state=self._state)

    def reset_arms(self) -> None:
        """
        Reset both arms, one after the other.

        If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

        If the arm isn't holding an object, it resets to its neutral position.
        """

        self.action = ResetArms(state=self._state)

    def navigate_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Navigate along a path to a target.

        See: [`NavigateTo`](navigate_to.md)

        :param target: The target object or position.
        """
        self.action = NavigateTo(target=target, collision_detection=self.collision_detection, state=self._state)

    def look_up(self, angle: float = 15) -> None:
        """
        Look upward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """

        self.rotate_head(axis="pitch", angle=-abs(angle), scale_duration=True)

    def look_down(self, angle: float = 15) -> None:
        """
        Look downward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """

        self.rotate_head(axis="pitch", angle=abs(angle), scale_duration=True)

    def look_at(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Look at a target object or position.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        """
        if target is None: 
            self.action = LookTo(target=self.home_look_at, duration=0.1, scale_duration=True, state=self._state)
        else:
            self.action = LookTo(target=target, duration=0.1, scale_duration=True, state=self._state)


    def place(self, position: Union[Dict[str, float], np.ndarray], arm: Arm, object_id: int, set_kinematic_state: bool = False, skip_drop: bool = False, object_rotate_angle: float = None) -> None:
        """
        Place a held target object at a target position.
        """
        self.action = Place(arm=arm, position=position, state=self._state, object_id = object_id, set_kinematic_state = set_kinematic_state, skip_drop = skip_drop, object_rotate_angle = object_rotate_angle)
        
    def reset_head(self, duration: float = 0.1, scale_duration: bool = True) -> None:
        """
        Reset the head to its neutral rotation.

        The head will continuously move over multiple `communicate()` calls until it is at its neutral rotation.

        :param duration: The duration of the motion in seconds.
        :param scale_duration: If True, `duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds.
        """

        super().reset_head(duration=duration, scale_duration=scale_duration)
    

    def _cache_static_data(self, resp: List[bytes]) -> None:
        """
        Cache static output data.

        :param resp: The response from the build.
        """
        org_commands = self.commands
        self.commands = []

        super()._cache_static_data(resp=resp)
        
        # fix the command for receiving images
        org_commands.extend([{"$type": "create_avatar",
                            "type": "A_Img_Caps_Kinematic",
                            "id": self.static.avatar_id},
                            {"$type": "set_pass_masks",
                            "pass_masks": ["_img", "_depth"],
                            "avatar_id": self.static.avatar_id},
                            {"$type": "parent_avatar_to_replicant",
                            "position": {"x": -1.1, "y": -0.6, "z": 0},
                            "avatar_id": self.static.avatar_id,
                            "id": self.replicant_id},
                            # {"$type": "teleport_avatar_to",
                            # "position": self.camera_position,
                            # "avatar_id": self.static.avatar_id}, 
                            # {"$type": "look_at_position",
                            # "position": self.camera_look_at,
                            # "avatar_id": self.static.avatar_id},
                            {"$type": "enable_image_sensor",
                            "enable": True,
                            "avatar_id": self.static.avatar_id},
                            {"$type": "set_img_pass_encoding",
                            "value": True},
                            {"$type": "send_images",
                            "frequency": "always"},
                            {"$type": "send_camera_matrices",
                            "frequency": "always"}])
        
        self.commands = org_commands