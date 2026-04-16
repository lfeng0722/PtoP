# ===========================================
# File: carla_controller.py
# ===========================================
import carla
import math

TIME_STEP = 0.05            # Simulation tick duration (s); 20 Hz.
TARGET_SPEED_KMH = 30       # Default cruise target speed (km/h).
LOOKAHEAD_DIST = 5.0        # Pure-pursuit lookahead distance (m).
KP_SPEED = 0.1              # Proportional gain for the longitudinal speed controller.
LANE_CHANGE_DISTANCE = 30.0 # Default distance to travel along the target lane during a lane change (m).
WAYPOINT_INTERVAL = 2.0     # Spacing between waypoints generated for lane-change paths (m).

class LaneKeepAndChangeController:
    """
    Lightweight lane-keeping and lane-changing controller for NPC vehicles.

    Supports three controller states:
        "LaneKeep"         — pure-pursuit lane following.
        "LaneChangeAccel"  — lane change with additional throttle.
        "LaneChangeDecel"  — lane change with reduced throttle.

    run_step() is called once per simulation tick and returns (control, signals),
    where signals is a dict of boolean completion flags for each maneuver type.
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()

        # Controller state: one of "LaneKeep", "LaneChangeAccel", "LaneChangeDecel".
        self.state = "LaneKeep"

        self.target_waypoints = []
        self.current_wp_idx = 0

        # Additional throttle/brake offsets applied on top of the PID output.
        self.extra_throttle = 0.0
        self.extra_brake = 0.0

    def run_step(self):
        """
        Execute one control step and return (control, signals).

        Returns:
            control: carla.VehicleControl for this tick.
            signals: dict with boolean flags:
                "lane_change_done":    True when a lane change maneuver completes.
                "acceleration_done":   True when extra_throttle has reached its maximum.
                "deceleration_done":   True when extra_throttle has reached its minimum.
                "brake_done":          True when extra_brake is at maximum.
                "release_brake_done":  True when extra_brake has been fully released.
        """
        control = carla.VehicleControl()

        # 1. Initialize completion flags for this tick.
        signals = {
            "lane_change_done": False,
            "acceleration_done": False,
            "deceleration_done": False,
            "brake_done": False,
            "release_brake_done": False,
        }

        # 2. Longitudinal control: base PID + state offset + extra_throttle / extra_brake.
        vel = self.vehicle.get_velocity()
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = speed_ms * 3.6
        speed_err = TARGET_SPEED_KMH - speed_kmh

        # Base PID throttle.
        throttle = KP_SPEED * speed_err

        # Adjust throttle based on lane-change state.
        if self.state == "LaneChangeAccel":
            throttle += 0.2
        elif self.state == "LaneChangeDecel":
            throttle -= 0.2

        # Apply extra_throttle / extra_brake offsets from the external API.
        throttle += self.extra_throttle
        brake = self.extra_brake

        # Clamp throttle and convert negative values to brake.
        if throttle > 1.0:
            throttle = 1.0
        elif throttle < 0.0:
            # Negative throttle becomes brake force.
            brake = max(brake, -throttle)
            throttle = 0.0
            if brake > 1.0:
                brake = 1.0

        control.throttle = throttle
        control.brake = brake

        # 3. Lateral control: lane keeping or lane changing based on current state.
        if self.state == "LaneKeep":
            steer = self._lane_keep_steer()
        elif self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            steer, done = self._lane_change_steer()
            if done:
                # Lane change complete; return to lane-keeping mode.
                self.state = "LaneKeep"
                self.target_waypoints.clear()
                self.current_wp_idx = 0
                signals["lane_change_done"] = True
                steer = 0.0
        else:
            steer = 0.0

        steer = max(-1.0, min(1.0, steer))
        control.steer = steer

        # 4. Update completion flags based on current extra throttle/brake values.
        if self.extra_throttle == 1.0:
            signals["acceleration_done"] = True
        if self.extra_throttle == -1.0:
            signals["deceleration_done"] = True
        if self.extra_brake == 1.0:
            signals["brake_done"] = True
        if self.extra_brake == 0.0:
            signals["release_brake_done"] = True

        return control, signals

    def _lane_keep_steer(self):
        transform = self.vehicle.get_transform()
        loc = transform.location
        yaw_rad = math.radians(transform.rotation.yaw)
        current_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if not current_wp:
            return 0.0

        dist_accum = 0.0
        step = 1.0
        temp_wp = current_wp
        while dist_accum < LOOKAHEAD_DIST:
            nxt = temp_wp.next(step)
            if not nxt:
                break
            temp_wp = nxt[0]
            dist_accum += step

        tx = temp_wp.transform.location.x
        ty = temp_wp.transform.location.y
        dx = tx - loc.x
        dy = ty - loc.y
        fx = dx * math.cos(-yaw_rad) - dy * math.sin(-yaw_rad)
        fy = dx * math.sin(-yaw_rad) + dy * math.cos(-yaw_rad)
        heading_err = math.atan2(fy, fx)
        return heading_err

    def _lane_change_steer(self):
        transform = self.vehicle.get_transform()
        loc = transform.location
        yaw_rad = math.radians(transform.rotation.yaw)

        # Reached the end of the waypoint list: lane change is done.
        if self.current_wp_idx >= len(self.target_waypoints):
            return (0.0, True)

        # Find the closest upcoming waypoint.
        min_dist = 1e9
        closest_idx = self.current_wp_idx
        for i in range(self.current_wp_idx, len(self.target_waypoints)):
            wp = self.target_waypoints[i]
            dx = wp.transform.location.x - loc.x
            dy = wp.transform.location.y - loc.y
            dsq = dx*dx + dy*dy
            if dsq < min_dist:
                min_dist = dsq
                closest_idx = i
        self.current_wp_idx = closest_idx

        # Close enough to the final waypoint: declare done.
        if self.current_wp_idx >= len(self.target_waypoints) - 1:
            return (0.0, True)

        # Look ahead by LOOKAHEAD_DIST along the waypoint path.
        look_idx = self.current_wp_idx
        dist_accum = 0.0
        tmp_wp = self.target_waypoints[look_idx]
        while look_idx < len(self.target_waypoints) - 1 and dist_accum < LOOKAHEAD_DIST:
            nxt_wp = self.target_waypoints[look_idx + 1]
            seg = tmp_wp.transform.location.distance(nxt_wp.transform.location)
            dist_accum += seg
            look_idx += 1
            tmp_wp = nxt_wp

        target_wp = self.target_waypoints[look_idx]
        tx = target_wp.transform.location.x
        ty = target_wp.transform.location.y
        dx = tx - loc.x
        dy = ty - loc.y
        fx = dx * math.cos(-yaw_rad) - dy * math.sin(-yaw_rad)
        fy = dx * math.sin(-yaw_rad) + dy * math.cos(-yaw_rad)
        heading_err = math.atan2(fy, fx)
        return (heading_err, False)

    # ========== External API: lane change, acceleration, braking ==========

    def request_lane_change_accel(self, direction: str, distance=LANE_CHANGE_DISTANCE):
        """
        Request an accelerating lane change in the given direction.

        Returns True if the request was accepted; False if a lane change is already in progress.
        """
        if self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            return False

        side_wp = self._get_side_waypoint(direction)
        if not side_wp:
            return False

        wps = self._gen_lanechange_wps(side_wp, distance, WAYPOINT_INTERVAL)
        if len(wps) < 2:
            print("[WARN] Insufficient waypoints for lane change; request rejected.")
            return False

        self.target_waypoints = wps
        self.current_wp_idx = 0
        self.state = "LaneChangeAccel"
        return True

    def request_lane_change_decel(self, direction: str, distance=LANE_CHANGE_DISTANCE):
        """
        Request a decelerating lane change in the given direction.

        Returns True if the request was accepted; False if a lane change is already in progress.
        """
        if self.state in ["LaneChangeAccel", "LaneChangeDecel"]:
            return False

        side_wp = self._get_side_waypoint(direction)
        if not side_wp:
            return False

        wps = self._gen_lanechange_wps(side_wp, distance, WAYPOINT_INTERVAL)
        if len(wps) < 2:
            print("[WARN] Insufficient waypoints for lane change; request rejected.")
            return False

        self.target_waypoints = wps
        self.current_wp_idx = 0
        self.state = "LaneChangeDecel"
        return True

    def _get_side_waypoint(self, direction: str):
        """
        Return the adjacent-lane waypoint in the given direction, or None if unavailable.

        Rejects the waypoint if the adjacent lane is a driving lane in the opposite direction
        (detected by a sign flip on lane_id).
        """
        loc = self.vehicle.get_location()
        current_wp = self.map.get_waypoint(loc, lane_type=carla.LaneType.Driving)
        if not current_wp:
            return None

        if direction == "left":
            side_wp = current_wp.get_left_lane()
        else:
            side_wp = current_wp.get_right_lane()

        if side_wp is None:
            return None
        if side_wp.lane_type != carla.LaneType.Driving:
            return None

        # Opposite-direction lane (sign flip on lane_id); reject.
        if side_wp.lane_id * current_wp.lane_id <= 0:
            return None

        return side_wp

    def _gen_lanechange_wps(self, start_wp, dist, interval):
        wps = []
        traveled = 0.0
        cur_wp = start_wp
        wps.append(cur_wp)

        while traveled < dist:
            nxt = cur_wp.next(interval)
            if not nxt:
                break
            cur_wp = nxt[0]
            wps.append(cur_wp)
            traveled += interval

        return wps

    def accelerate(self, val=0.1):
        """
        Increase extra throttle by val.

        Returns True when extra_throttle has reached its upper limit (1.0).
        """
        self.extra_throttle += val
        if self.extra_throttle > 1.0:
            self.extra_throttle = 1.0
        return (self.extra_throttle >= 1.0)

    def decelerate(self, val=0.1):
        """
        Decrease extra throttle by val.

        Returns True when extra_throttle has reached its lower limit (-1.0).
        """
        self.extra_throttle -= val
        if self.extra_throttle < -1.0:
            self.extra_throttle = -1.0
        return (self.extra_throttle <= -1.0)

    def brake(self, val=0.1):
        """
        Increase extra brake by val.

        Returns True when extra_brake has reached its upper limit (1.0).
        """
        self.extra_brake += val
        if self.extra_brake > 1.0:
            self.extra_brake = 1.0
        return (self.extra_brake >= 1.0)

    def release_brake(self, val=0.1):
        """
        Decrease extra brake by val.

        Returns True when extra_brake has been fully released (0.0).
        """
        self.extra_brake -= val
        if self.extra_brake < 0.0:
            self.extra_brake = 0.0
        return (self.extra_brake <= 0.0)
