# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed


class VehiclePIDController:
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, offset, **args_lateral)


    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint/transform at a given target_speed.

        :param target_speed: desired vehicle speed (m/s), positive magnitude
        :param waypoint: target as a carla.Waypoint (has .transform) OR carla.Transform
        :return: carla.VehicleControl
        """
        # --- Resolve target transform (accept Waypoint or Transform) ---
        target_tf = waypoint.transform if hasattr(waypoint, "transform") else waypoint
        assert isinstance(target_tf, carla.Transform), "waypoint must be Waypoint or Transform"

        veh_tf  = self._vehicle.get_transform()
        veh_loc = veh_tf.location
        veh_rot = veh_tf.rotation

        # --- Decide forward vs reverse: projection onto vehicle forward vector ---
        # Forward unit vector in world XY from vehicle yaw
        fyaw = math.radians(veh_rot.yaw)
        fx, fy = math.cos(fyaw), math.sin(fyaw)

        dx = target_tf.location.x - veh_loc.x
        dy = target_tf.location.y - veh_loc.y
        proj = dx * fx + dy * fy   # > 0 => target is ahead; < 0 => behind

        reverse_mode = (proj < 0.0)

        # throttle scaling when reversing (gentler)
        reverse_speed_scale = getattr(self, "reverse_speed_scale", 0.7)
        fwd_speed = target_speed
        rev_speed = max(0.0, target_speed) * reverse_speed_scale * np.clip(abs(proj), 0.5, 1.0) if abs(proj) > 0.1 and not self._lat_controller.brake else 0

        # --- Longitudinal control ---
        speed_cmd = fwd_speed if not reverse_mode else rev_speed
        acceleration = self._lon_controller.run_step(speed_cmd)

        # --- Lateral control (heading/crosstrack) ---
        current_steering = self._lat_controller.run_step(target_tf, reverse_mode)
        if reverse_mode:
            # Flip steering when backing up (right-turn while reversing moves nose left)
            current_steering = -current_steering

        control = carla.VehicleControl()

        # Throttle/brake from "acceleration"
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering rate limiting 
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        # Clamp steering to max_steer
        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False

        # --- Reverse gear handling ---
        control.reverse = bool(reverse_mode)

        self.past_steering = steering
        return control

    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLateralController"""
        self._lat_controller.change_parameters(**args_lateral)

    def set_offset(self, offset):
        """Changes the offset"""
        self._lat_controller.set_offset(offset)


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self.brake = False
        self._e_buffer = deque(maxlen=10)

    def run_step(self, waypoint, reverse_mode):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoint.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform(), reverse_mode)

    def set_offset(self, offset):
        """Changes the offset"""
        self._offset = offset

    def _pid_control(self, target_wp, vehicle_transform, reverse_mode):
        """
        Lateral PID controller: returns steering in [-1, 1].

        Respects:
        - self._offset (meters): lateral shift along waypoint right-vector
        - self._k_p, self._k_i, self._k_d
        - self._dt
        - optional: self._use_wp_heading (bool), self._alpha in [0,1]
            _alpha = 0.0 -> pure vector-to-waypoint (default)
            _alpha = 1.0 -> pure waypoint heading
            in-between -> blend
        - optional: self._i_clamp (float) to limit integral term magnitude
        """

        # --- 1) Get a carla.Transform ---
        tf = getattr(target_wp, "transform", target_wp)

        # --- 2) Ego forward unit vector (XY) ---
        v_fwd = vehicle_transform.get_forward_vector()
        v = np.array([v_fwd.x, v_fwd.y], dtype=np.float64)
        v_head = v / (np.linalg.norm(v) + 1e-9)

        vel = self._vehicle.get_velocity()
        vel_xy = np.array([vel.x, vel.y], dtype=np.float64)
        speed = np.linalg.norm(vel_xy)

        # --- 3) Target location with optional lateral offset along waypoint right-vector ---
        if getattr(self, "_offset", 0.0) != 0.0:
            r_vec = tf.get_right_vector()
            w_loc = tf.location + carla.Location(
                x=self._offset * r_vec.x,
                y=self._offset * r_vec.y
            )
        else:
            w_loc = target_wp.location

        # --- 4) Build candidate target directions (unit vectors in XY) ---
        # 4a) Vector from ego to (possibly offset) waypoint
        dx = w_loc.x - vehicle_transform.location.x
        dy = w_loc.y - vehicle_transform.location.y
        vec_to_wp = np.array([dx, dy], dtype=np.float64)
        vec_to_wp /= (np.linalg.norm(vec_to_wp) + 1e-9)

        if vel_xy @ vec_to_wp < 0 and speed > 0.05:
            self.brake = True
        else:
            self.brake = False

        # 4b) Waypoint heading
        yaw_wp = math.radians(target_wp.rotation.yaw) if not reverse_mode else math.radians(target_wp.rotation.yaw)
        
        wp_heading = np.array([math.cos(yaw_wp), math.sin(yaw_wp)], dtype=np.float64)

        alpha = 0.4

        if not reverse_mode:
            e_aligning = self.signed_angle(v_head, wp_heading)
            e_tracking = self.signed_angle(v_head, vec_to_wp)
            e = alpha * e_aligning + (1 - alpha) * e_tracking
        else:
            e_aligning = self.signed_angle(v_head, wp_heading)
            e_tracking = -self.signed_angle(-v_head, vec_to_wp)

            e = alpha * e_aligning + 2 * (1 - alpha) * e_tracking / (speed + 1e-3)

        # --- 5) PID terms ---
        dt = max(getattr(self, "_dt", 0.05), 1e-3)
        self._e_buffer.append(e)
        if len(self._e_buffer) >= 2:
            de = (self._e_buffer[-1] - self._e_buffer[-2]) / dt
            ie = sum(self._e_buffer) * dt
        else:
            de, ie = 0.0, 0.0

        # Optional integral clamp to avoid windup (set self._i_clamp, e.g. 0.5)
        i_clamp = getattr(self, "_i_clamp", None)
        if i_clamp is not None:
            ie = float(np.clip(ie, -abs(i_clamp), abs(i_clamp)))

        steer = (self._k_p * e) + (self._k_d * de) + (self._k_i * ie)
        return float(np.clip(steer, -1.0, 1.0))

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

    def signed_angle(self, v, t):
        """ Signed angle, v to t """
        cross = v[0] * t[1] - v[1] * t[0]
        dot     = v[0] * t[0] + v[1] * t[1]
        e = math.atan2(cross, dot)  # radians in (-pi, pi]
        return e
        