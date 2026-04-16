import requests
import carla
from carla_controller import LaneKeepAndChangeController
import random
from websocket import create_connection, WebSocketException
import json
import threading
import math
import logging
import time
import numpy as np
import collections
import os

max_search_distance_for_destination = 200  # Maximum BFS search distance when locating the destination waypoint (meters).
step_dist_for_destination = 2.0            # BFS step size for destination search (meters).
max_search_distance_for_spawns = 50.0      # Search radius when selecting multi-lane spawn points (meters).
step_for_spawns = 1.0

log = logging.getLogger(__name__)


EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # Minimum ego closing speed along collision normal to assign fault (m/s).
EGO_FAULT_RATIO          = 0.60   # Minimum fraction of total closing speed attributable to ego for fault assignment.
IMPULSE_MIN              = 400.0  # Minimum collision impulse magnitude; below this the event is treated as a graze.
REAR_END_BONUS           = 0.05   # Reduced fault ratio threshold when the ego appears to rear-end the other actor.

# -------- Vector / speed / unit-vector / dot-product helpers --------
def _vec_norm(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def _spd_and_vec(actor):
    v = actor.get_velocity()
    return _vec_norm(v), v

def _unit_vec(a: "carla.Location", b: "carla.Location"):
    dx, dy, dz = (b.x - a.x), (b.y - a.y), (b.z - a.z)
    n = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
    return carla.Vector3D(dx/n, dy/n, dz/n)

def _dot(a: "carla.Vector3D", b: "carla.Vector3D"):
    return a.x*b.x + a.y*b.y + a.z*b.z

def _ego_local_sd(ego_tf: "carla.Transform", loc: "carla.Location"):
    yaw = math.radians(ego_tf.rotation.yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    s =  dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d

def _assign_blame_ego(ego: "carla.Vehicle",
                      other: "carla.Actor",
                      world_map: "carla.Map",
                      event_normal_impulse: "carla.Vector3D") -> (bool, str):
    “””
    Attribute collision fault to the ego vehicle.

    Returns: (ego_fault: bool, why: str)

    Logic:
      1) Compute unit vector n from ego to other; decompose each vehicle's velocity along n.
      2) c_ego   = ego closing speed toward other (component along +n).
         c_other = other closing speed toward ego (component along -n).
      3) r = c_ego / (c_ego + c_other).
      4) Assign ego fault if: impulse >= IMPULSE_MIN, c_ego >= threshold, and r >= ratio threshold
         (threshold is slightly relaxed for rear-end scenarios).
    “””
    try:
        # Unit vector from ego toward the other actor.
        loc_e = ego.get_transform().location
        loc_o = other.get_transform().location
        n = _unit_vec(loc_e, loc_o)

        # Closing speed components along the ego→other axis.
        spd_e, v_e = _spd_and_vec(ego)
        if hasattr(other, “get_velocity”):
            spd_o, v_o = _spd_and_vec(other)
        else:
            spd_o, v_o = 0.0, carla.Vector3D(0.0, 0.0, 0.0)

        # Ego's closing speed toward other (positive = approaching other).
        c_ego   = max(0.0, _dot(v_e, n))
        # Other's closing speed toward ego: other moves along -n, equivalent to -(v_o · n).
        c_other = max(0.0, -_dot(v_o, n))

        # Collision impulse magnitude.
        J = _vec_norm(event_normal_impulse)

        # Pose relationship for rear-end / side-swipe discrimination.
        s_rel, d_rel = _ego_local_sd(ego.get_transform(), loc_o)  # s > 0: other is ahead of ego.
        rear_end_like = (s_rel > 0.0 and c_ego > c_other)  # Ego appears to rear-end the vehicle ahead.

        # Fault ratio and adaptive threshold.
        r = c_ego / (c_ego + c_other + 1e-9)
        thr = EGO_FAULT_RATIO - (REAR_END_BONUS if rear_end_like else 0.0)

        # Assign fault.
        if J >= IMPULSE_MIN and c_ego >= EGO_FAULT_CLOSE_SPEED_MIN and r >= thr:
            reason = f"ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
            return True, reason
        else:
            reason = f"non_ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
            return False, reason
    except Exception as e:
        # On error, conservatively do not assign fault to ego.
        return False, f"non_ego_fault: exception {e}"

def fetch_localization_variable(url="http://127.0.0.1:5000/var"):
    """
    Fetch the latest localization data from the listener Flask endpoint via HTTP GET.

    Args:
        url: Endpoint address. Defaults to 127.0.0.1:5000/var (local listener).

    Returns:
        Dict with localization data in JSON format, or None on failure.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print("Failed to fetch localization data:", e)
        return None

def distance(loc1, loc2):
    """Compute the 2D Euclidean distance between two CARLA Location objects."""
    return math.sqrt(
        (loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2
    )

SPAWN_GAP_VEH = 4.5   # Minimum required spacing between any two NPC vehicles (meters).
SPAWN_GAP_PED = 2.5   # Minimum required spacing between any two pedestrians (meters).
SPAWN_GAP_EGO = 7.0   # Minimum required spacing between any NPC and the ego vehicle (meters).

def _gap_ok(tf, accepted_tfs, min_gap):
    """Return True if tf is at least min_gap meters from every Transform in accepted_tfs."""
    for t in accepted_tfs:
        dx = tf.location.x - t.location.x
        dy = tf.location.y - t.location.y
        if math.hypot(dx, dy) < float(min_gap):
            return False
    return True
# ========================================

class MultiVehicleDemo:
    """
    Manages the ego vehicle, NPC vehicles, pedestrians, and their sensors for one test episode.

    Responsibilities:
      1) Spawn the ego vehicle and N NPC vehicles with LaneKeepAndChangeController controllers.
      2) Attach collision sensors to all vehicles.
         - Ego collision events: record the collision regardless of fault attribution.
         - NPC collision events: apply emergency braking without ending the episode.
      3) tick() runs one simulation step and returns
         (signals_list, ego_collision, all_collision, ego_cross_solid_line, ego_run_red_light).
      4) set_destination() uses a BFS over road waypoints to find the farthest same-direction
         point and stores it as self.ego_destination.
      5) get_controller(idx) returns the LaneKeepAndChangeController for the vehicle at index idx.
    """

    def __init__(self, world, external_ads, websocket_url="ws://localhost:8888/websocket",
                 gps_offset=carla.Vector3D(x=1.0, y=0.0, z=0.5)):
        self.world = world
        self.population_size = 10
        self.map = world.get_map()
        self.ego_vehicle = None
        self.multi_vehicle_collision_count = 0
        self.vehicles = []            # Spawned NPC vehicles.
        self.controllers = None       # One LaneKeepAndChangeController per NPC vehicle.
        self.url = websocket_url
        self.gps_offset = gps_offset
        self.ws = None
        self.vehicle_num = None
        self.ws_thread = None
        self.ws_running = False
        self.ws_receive_buffer = []
        self.ego_spawning_point = None
        self.ego_destination = None   # Set by set_destination().
        self.collision = False
        self.external_ads = external_ads
        self.count = 0
        self.turn_on = False
        self.modules = [
            'Localization',
            'Routing',
            'Prediction',
            'Planning',
            'Control'
        ]
        self.side_collision_count_vehicle = 0  # Side collision counter.
        self.rear_collision_count_vehicle = 0  # Rear-end collision counter.
        self.collision_count_obj = 0

        # Flag indicating that the ego vehicle was involved in a collision.
        self.ego_collision = False

        # Map bounding box (min_x, max_x, min_y, max_y).
        self.map_bounds = self._compute_map_bounds()

        # List of active collision sensor actors.
        self.collision_sensors = []

        # ----- Lane-invasion detection -----
        self.ego_cross_solid_line = False  # True if ego crossed a solid lane marking.
        self.lane_invasion_sensor_ego = None

        # ----- Red-light detection -----
        self.ego_run_red_light = False  # True if a red-light violation was detected.

        if self.external_ads:
            self._connect_websocket()

    # ========== Helper methods ==========

    def _is_npc_rear_end(self, ego: “carla.Vehicle”, npc: “carla.Vehicle”) -> bool:
        “””
        Return True if the NPC appears to be rear-ending the ego vehicle.

        Conditions (all must hold):
          - NPC is behind ego (s_rel < -0.5 in ego's local frame).
          - Lateral offset is within 0.4 * lane_width (roughly the same lane).
          - Heading difference <= 35 degrees (same direction of travel).
          - NPC is closing on ego (dv_f > 0.5 m/s).
        “””
        try:
            ego_tf = ego.get_transform()
            npc_tf = npc.get_transform()

            # Relative longitudinal and lateral positions in the ego frame.
            s_rel, d_rel = self._ego_local_sd(ego_tf, npc_tf.location)

            # Ego forward unit vector.
            yaw = math.radians(ego_tf.rotation.yaw)
            fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

            v_e = ego.get_velocity()
            v_n = npc.get_velocity()
            v_e_f = v_e.x * fwd.x + v_e.y * fwd.y + v_e.z * fwd.z
            v_n_f = v_n.x * fwd.x + v_n.y * fwd.y + v_n.z * fwd.z
            dv_f  = v_n_f - v_e_f  # NPC forward speed relative to ego (>0 = closing from behind).

            # Similar heading (same travel direction) and small lateral offset (same lane).
            dyaw = abs(((npc_tf.rotation.yaw - ego_tf.rotation.yaw + 180.0) % 360.0) - 180.0)
            lane_w = 3.5  # Default lane width (m); overridden by waypoint data if available.
            try:
                wp = self.map.get_waypoint(ego_tf.location)
                if wp and hasattr(wp, "lane_width"):
                    lane_w = float(wp.lane_width)
            except Exception:
                pass

            return (s_rel < -0.5) and (abs(d_rel) <= 0.4 * lane_w) and (dyaw <= 35.0) and (dv_f > 0.5)
        except Exception:
            return False

    def _connect_websocket(self):
        try:
            self.ws = create_connection(self.url)
            self.ws_running = True
            print(f"[INFO] Connected to WebSocket server: {self.url}")
            # Start a background thread to receive incoming WebSocket messages.
            self.ws_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.ws_thread.start()
        except WebSocketException as e:
            print(f"[ERROR] Could not connect to WebSocket server: {e}")
            self.ws = None

    def _receive_messages(self):
        while self.ws_running:
            try:
                result = self.ws.recv()
                if result:
                    self.ws_receive_buffer.append(result)
            except WebSocketException as e:
                print(f"[ERROR] WebSocket receive error: {e}")
                self.ws_running = False
            except Exception as e:
                print(f"[ERROR] Unexpected error in WebSocket receive thread: {e}")
                self.ws_running = False

    def _compute_map_bounds(self):
        """
        Compute the (min_x, max_x, min_y, max_y) bounding box of the map via waypoint sampling.
        """
        wps = self.map.generate_waypoints(2.0)
        if not wps:
            print("[WARN] generate_waypoints returned empty; map may have no data.")
            return (0, 0, 0, 0)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for wp in wps:
            loc = wp.transform.location
            if loc.x < min_x: min_x = loc.x
            if loc.x > max_x: max_x = loc.x
            if loc.y < min_y: min_y = loc.y
            if loc.y > max_y: max_y = loc.y
        print(f"[INFO] Map bounds: x=({min_x:.1f}, {max_x:.1f}), y=({min_y:.1f}, {max_y:.1f})")
        return (min_x, max_x, min_y, max_y)

    def get_map_bounds(self):
        return self.map_bounds

    # ========== Vehicle and pedestrian spawn logic ==========

    def setup_vehicles(self, scenario_conf):
        “””
        Spawn all actors for the scenario.

        1) Spawn the ego vehicle at scenario_conf['ego_transform'].
        2) Spawn each NPC in scenario_conf['surrounding_info'] order (vehicles, bicycles, pedestrians).
           - On spawn collision: resample along the lane (forward/backward and lateral shifts) until success.
           - Pedestrian failure: resample from the navigation mesh until success.
        3) Write the final successful spawn transform back into scenario_conf to keep
           the scenario representation consistent with the actual simulation state.
        “””
        world = self.world
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        self.vehicle_num = int(scenario_conf["vehicle_num"])
        self.controllers = [None] * self.vehicle_num

        # ------- Parse surrounding_info (supports both list and dict-of-lists formats) -------
        surrounding = scenario_conf["surrounding_info"]

        # Two supported encodings: list[{"transform","type"}] or dict{"transform":[...], "type":[...]}
        def _get_item(i):
            if isinstance(surrounding, list):
                return surrounding[i]["transform"], str(surrounding[i]["type"]).lower()
            else:
                return surrounding["transform"][i], str(surrounding["type"][i]).lower()

        def _set_item_transform(i, new_tf):
            if isinstance(surrounding, list):
                surrounding[i]["transform"] = new_tf
            else:
                surrounding["transform"][i] = new_tf

        n_to_spawn = min(self.vehicle_num,
                         len(surrounding) if isinstance(surrounding, list) else len(surrounding["transform"]))

        # ------- EGO -------
        self.ego_spawning_point = scenario_conf["ego_transform"]
        self.ego_vehicle = None
        if not getattr(self, "external_ads", False):
            try:
                bp_ego = blueprint_library.find("vehicle.tesla.model3")
            except Exception:
                veh_bps = blueprint_library.filter("vehicle.*")
                four_wheels = [bp for bp in veh_bps if bp.has_attribute("number_of_wheels")
                               and int(bp.get_attribute("number_of_wheels").as_int()) == 4]
                bp_ego = random.choice(four_wheels if four_wheels else veh_bps)
            if bp_ego.has_attribute("color"):
                bp_ego.set_attribute("color", "0,0,255")
            self.ego_vehicle = world.try_spawn_actor(bp_ego, self.ego_spawning_point)
        else:
            # External ADS mode: locate the existing mkz_2017 vehicle and teleport it to ego_transform.
            all_actors = world.get_actors()
            candidate_vehicles = all_actors.filter("vehicle.*")
            for v in candidate_vehicles:
                if "mkz_2017" in v.type_id:
                    self.ego_vehicle = v
                    break
            if not self.ego_vehicle:
                print("[ERROR] Could not find 'mkz_2017' vehicle to use as ego.")
                return False
            self.ego_vehicle.set_transform(self.ego_spawning_point)

        if not self.ego_vehicle:
            print("[ERROR] Ego vehicle spawn failed.")
            return False

        # ------- Blueprint pools -------
        veh_bps_all = blueprint_library.filter("vehicle.*")
        car_bps = blueprint_library.filter("vehicle.tesla.model3") or veh_bps_all
        bike_bps = [bp for bp in veh_bps_all if ("bicycle" in bp.id.lower() or "bike" in bp.id.lower())]
        walker_bps = blueprint_library.filter("walker.pedestrian.*")

        def _pick(pool, fallback):
            if pool: return random.choice(pool)
            if fallback: return random.choice(fallback)
            return random.choice(veh_bps_all)

        # ------- Geometry / lane helpers -------
        def _project_to_lane(tf, clip_ratio=0.45):
            wp = world_map.get_waypoint(tf.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if not wp:
                return tf, None
            lane_w = float(getattr(wp, "lane_width", 3.5))
            # Preserve relative lateral offset but clip to clip_ratio * lane_width.
            center = wp.transform.location
            right = wp.transform.get_right_vector()
            dv = carla.Vector3D(tf.location.x - center.x, tf.location.y - center.y, tf.location.z - center.z)
            dlat = dv.x * right.x + dv.y * right.y + dv.z * right.z
            d_clip = float(np.clip(dlat, -clip_ratio * lane_w, clip_ratio * lane_w))
            loc = carla.Location(center.x + d_clip * right.x,
                                 center.y + d_clip * right.y,
                                 tf.location.z)
            # Align yaw to the lane direction for a stable spawn orientation.
            yaw = wp.transform.rotation.yaw
            return carla.Transform(loc, carla.Rotation(pitch=tf.rotation.pitch, yaw=yaw, roll=tf.rotation.roll)), lane_w

        def _lane_shift_candidates(tf0, max_forward=18.0, step_s=2.0, step_d=0.75, d_mul=0.45):
            """Sample candidate spawn poses along the lane center (±s) and laterally (±d), nearest first."""
            base_wp = world_map.get_waypoint(tf0.location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if not base_wp:
                return [tf0]

            # Build longitudinal offset sequence: 0, +2, -2, +4, -4, ...
            s_vals = [0.0]
            k = int(max_forward // step_s)
            for i in range(1, k + 1):
                s_vals += [i * step_s, -i * step_s]

            # Build lateral offset sequence: 0, +0.75, -0.75, +1.5, -1.5, ...
            lane_w = float(getattr(base_wp, "lane_width", 3.5))
            d_max = d_mul * lane_w
            d_vals = [0.0]
            kd = max(1, int(d_max // step_d))
            for i in range(1, kd + 1):
                d_vals += [i * step_d, -i * step_d]

            cands = []
            for s in s_vals:
                # Never call next(0.0) or previous(0.0); use the current waypoint directly for s == 0.
                if s > 0.0:
                    wps = base_wp.next(s)
                elif s < 0.0:
                    wps = base_wp.previous(-s)
                else:
                    wps = [base_wp]  # s == 0: use the current waypoint directly.

                if not wps:
                    continue
                wp = wps[0]
                center = wp.transform.location
                right = wp.transform.get_right_vector()
                yaw = wp.transform.rotation.yaw

                for d in d_vals:
                    loc = carla.Location(center.x + d * right.x,
                                         center.y + d * right.y,
                                         tf0.location.z)
                    cands.append(carla.Transform(
                        loc,
                        carla.Rotation(pitch=tf0.rotation.pitch, yaw=yaw, roll=tf0.rotation.roll)
                    ))
            return cands

        def _tick_flush():
            try:
                world.tick()
            except Exception:
                world.wait_for_tick()
            time.sleep(0.01)

        # ------- Containers -------
        if not hasattr(self, "vehicles"): self.vehicles = []
        if not hasattr(self, "pedestrians"): self.pedestrians = []

        # ------- Spawn NPCs one by one (resample until each succeeds) -------
        spawned_vehicle_count = 0
        spawned_ped_count = 0

        print('vehicle number (requested):', self.vehicle_num)

        # Track successfully placed transforms (separate lists for vehicles and pedestrians).
        veh_tfs = []
        ped_tfs = []

        for i in range(n_to_spawn):
            init_tf, npc_type = _get_item(i)
            actor = None

            try:
                if npc_type == "pedestrian":
                    if not walker_bps:
                        print(f"[WARN] No pedestrian blueprint available; skipping NPC[{i}].")
                        continue
                    bp = random.choice(walker_bps)

                    # Try the original position first (if it satisfies the minimum gap).
                    if self.ego_vehicle:
                        ego_loc_now = self.ego_vehicle.get_transform().location
                        if math.hypot(init_tf.location.x - ego_loc_now.x,
                                      init_tf.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                            actor = None
                        elif not _gap_ok(init_tf, ped_tfs, SPAWN_GAP_PED):
                            actor = None
                        else:
                            actor = world.try_spawn_actor(bp, init_tf)
                    else:
                        # Edge case: ego not yet spawned; only check gap against already-placed pedestrians.
                        if _gap_ok(init_tf, ped_tfs, SPAWN_GAP_PED):
                            actor = world.try_spawn_actor(bp, init_tf)

                    if not actor:
                        # Resample from the navigation mesh until a valid spawn is found.
                        attempts = 0
                        while actor is None:
                            loc = world.get_random_location_from_navigation()
                            if loc is None:
                                attempts += 1
                                if attempts % 10 == 0: _tick_flush()
                                continue
                            tf_try = carla.Transform(loc, init_tf.rotation)

                            # Distance constraints against ego and already-placed pedestrians.
                            ok_ego = True
                            if self.ego_vehicle:
                                ego_loc_now = self.ego_vehicle.get_transform().location
                                if math.hypot(tf_try.location.x - ego_loc_now.x,
                                              tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                    ok_ego = False
                            if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                actor = world.try_spawn_actor(bp, tf_try)
                                attempts += 1
                                if actor:
                                    _set_item_transform(i, tf_try)
                                    ped_tfs.append(tf_try)
                                    break
                            else:
                                attempts += 1

                            if attempts % 10 == 0:
                                _tick_flush()
                    else:
                        # Original position succeeded; also write back for consistency.
                        _set_item_transform(i, init_tf)
                        ped_tfs.append(init_tf)

                    if actor:
                        self.pedestrians.append(actor)
                        spawned_ped_count += 1
                    else:
                        # Fallback: keep retrying with the navigation mesh until successful.
                        while actor is None:
                            loc = world.get_random_location_from_navigation()
                            if loc:
                                tf_try = carla.Transform(loc, init_tf.rotation)
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        self.pedestrians.append(actor)
                                        ped_tfs.append(tf_try)
                                        spawned_ped_count += 1
                                        break
                            _tick_flush()

                else:
                    # Vehicle / bicycle (falls back to car); resample near a lane until successful.
                    if npc_type == "bicycle":
                        bp = _pick(bike_bps, car_bps)
                    elif npc_type == "car":
                        bp = _pick(car_bps, veh_bps_all)
                    else:
                        bp = _pick(car_bps, veh_bps_all)

                    # Project init_tf to the lane centerline first.
                    tf0, _ = _project_to_lane(init_tf)

                    # Try the projected position first (if it satisfies the minimum gap).
                    can_try = True
                    if self.ego_vehicle:
                        ego_loc_now = self.ego_vehicle.get_transform().location
                        if math.hypot(tf0.location.x - ego_loc_now.x,
                                      tf0.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                            can_try = False
                    if can_try and not _gap_ok(tf0, veh_tfs, SPAWN_GAP_VEH):
                        can_try = False
                    actor = world.try_spawn_actor(bp, tf0) if can_try else None

                    if actor:
                        _set_item_transform(i, tf0)
                        veh_tfs.append(tf0)
                    else:
                        # Generate candidates along the lane and keep trying; widen search range each round.
                        attempts = 0
                        max_forward = 18.0
                        while actor is None:
                            candidates = _lane_shift_candidates(tf0, max_forward=max_forward,
                                                                step_s=2.0, step_d=0.75, d_mul=0.45)
                            for tf_try in candidates:
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    attempts += 1
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        veh_tfs.append(tf_try)
                                        break
                                else:
                                    attempts += 1
                                if attempts % 15 == 0:
                                    _tick_flush()
                            if actor:
                                break
                            # Widen the search range for the next round.
                            max_forward = min(max_forward + 12.0, 60.0)
                            if attempts > 200:
                                # Fallback: draw randomly from global spawn points until successful.
                                sps = world_map.get_spawn_points()
                                random.shuffle(sps)
                                for tf_try in sps:
                                    ok_ego = True
                                    if self.ego_vehicle:
                                        ego_loc_now = self.ego_vehicle.get_transform().location
                                        if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                      tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                            ok_ego = False
                                    if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                        actor = world.try_spawn_actor(bp, tf_try)
                                        attempts += 1
                                        if actor:
                                            _set_item_transform(i, tf_try)
                                            veh_tfs.append(tf_try)
                                            break
                                    else:
                                        attempts += 1
                                    if attempts % 15 == 0:
                                        _tick_flush()
                            if attempts > 400 and actor is None:
                                # Keep retrying until successful; flush every 30 attempts.
                                tf_try = random.choice(world_map.get_spawn_points())
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        veh_tfs.append(tf_try)
                                        break
                                _tick_flush()

                    if actor:
                        self.vehicles.append(actor)
                        spawned_vehicle_count += 1

            except Exception as e:
                print(f”[ERROR] Failed to spawn NPC[{i}]: {e}”)
                # Emergency fallback: retry until successful.
                if npc_type == “pedestrian” and walker_bps:
                    bp = random.choice(walker_bps)
                    while True:
                        loc = world.get_random_location_from_navigation()
                        if loc:
                            tf_try = carla.Transform(loc, init_tf.rotation)
                            ok_ego = True
                            if self.ego_vehicle:
                                ego_loc_now = self.ego_vehicle.get_transform().location
                                if math.hypot(tf_try.location.x - ego_loc_now.x,
                                              tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                    ok_ego = False
                            if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                a2 = world.try_spawn_actor(bp, tf_try)
                                if a2:
                                    _set_item_transform(i, tf_try)
                                    self.pedestrians.append(a2)
                                    ped_tfs.append(tf_try)
                                    spawned_ped_count += 1
                                    break
                        _tick_flush()
                else:
                    bp = _pick(car_bps, veh_bps_all)
                    tf0, _ = _project_to_lane(init_tf)
                    while True:
                        # Random global spawn point.
                        tf_try = random.choice(world_map.get_spawn_points())
                        ok_ego = True
                        if self.ego_vehicle:
                            ego_loc_now = self.ego_vehicle.get_transform().location
                            if math.hypot(tf_try.location.x - ego_loc_now.x,
                                          tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                ok_ego = False
                        if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                            a2 = world.try_spawn_actor(bp, tf_try)
                            if a2:
                                _set_item_transform(i, tf_try)
                                self.vehicles.append(a2)
                                veh_tfs.append(tf_try)
                                spawned_vehicle_count += 1
                                break
                        _tick_flush()

        print("spawned vehicles (vehicle.*):", spawned_vehicle_count)
        print("spawned pedestrians:", spawned_ped_count)

        # Attach controllers to all spawned vehicles (not pedestrians).

        for i, v in enumerate(self.vehicles):
            try:
                self.controllers[i] = LaneKeepAndChangeController(v)
            except Exception as e:
                print(f”[WARN] Failed to create controller for veh[{i}] id={v.id}: {e}”)

        return True

    def _is_valid_side_lane(self, wp, side_wp):
        """
        Return True if side_wp is a Driving lane with the same direction as wp
        (same sign on lane_id indicates same travel direction).
        """
        if not side_wp:
            return False
        if side_wp.lane_type != carla.LaneType.Driving:
            return False
        if wp.lane_id * side_wp.lane_id <= 0:
            return False
        return True

    def setup_vehicles_with_collision(self, scenario_conf):
        """
        Public entry point: spawn all NPC vehicles/pedestrians and attach collision sensors.
        Calls setup_vehicles first; if successful, attaches collision sensors via
        _setup_collision_sensors.
        """
        success = self.setup_vehicles(scenario_conf)
        if success:
            self._setup_collision_sensors()
        return success

    # ========== Collision sensor logic + LaneInvasion sensor logic ==========

    def _setup_collision_sensors(self):
        """
        Attach collision sensors to the ego vehicle and all NPC vehicles.
        Also attaches a lane-invasion sensor to the ego vehicle.
        """
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # Ego collision sensor.
        if self.ego_vehicle:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor_ego = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
            # Pass the sensor reference into the callback so it can be cleaned up on trigger.
            sensor_ego.listen(lambda event, v=self.ego_vehicle, s=sensor_ego: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor_ego)
            print(f"[INFO] Ego vehicle {self.ego_vehicle.id}: collision sensor attached (id={sensor_ego.id}).")

            # Attach a lane-invasion sensor to the ego vehicle.
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.lane_invasion_sensor_ego = self.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor_ego.listen(self.lane_invasion_callback)
            print(f"[INFO] Ego vehicle {self.ego_vehicle.id}: lane-invasion sensor attached (id={self.lane_invasion_sensor_ego.id}).")

        # Collision sensors for all NPC vehicles.
        for veh in self.vehicles:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=veh)
            # Pass the sensor reference into the callback so it can be cleaned up on trigger.
            sensor.listen(lambda event, v=veh, s=sensor: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor)
            print(f"[INFO] Vehicle {veh.id}: collision sensor attached (id={sensor.id}).")

    def lane_invasion_callback(self, event):
        """
        Triggered when the ego vehicle crosses a lane marking.
        Sets ego_cross_solid_line if any crossed marking is a solid line type.
        """
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid
            ]:
                # A solid marking type indicates the ego vehicle crossed a solid lane line.
                self.ego_cross_solid_line = True
                print("[INFO] Ego vehicle crossed a solid lane marking.")
                break

    # ==================== Fault attribution block ====================

    # Thresholds (tunable).
    EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # Minimum ego closing speed along the collision normal (m/s).
    EGO_FAULT_RATIO          = 0.60   # Threshold for ego's share of total closing speed.
    IMPULSE_MIN              = 400.0  # Minimum collision impulse magnitude to consider fault.
    REAR_END_BONUS           = 0.05   # Relaxed ratio threshold applied in rear-end scenarios.

    # Static helpers used internally for blame attribution.
    @staticmethod
    def _vec_norm(v: "carla.Vector3D") -> float:
        return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    @staticmethod
    def _spd_and_vec(actor):
        v = actor.get_velocity()
        return MultiVehicleDemo._vec_norm(v), v

    @staticmethod
    def _unit_vec(a: "carla.Location", b: "carla.Location") -> "carla.Vector3D":
        dx, dy, dz = (b.x - a.x), (b.y - a.y), (b.z - a.z)
        n = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
        return carla.Vector3D(dx/n, dy/n, dz/n)

    @staticmethod
    def _dot(a: "carla.Vector3D", b: "carla.Vector3D") -> float:
        return a.x*b.x + a.y*b.y + a.z*b.z

    @staticmethod
    def _ego_local_sd(ego_tf: "carla.Transform", loc: "carla.Location"):
        yaw = math.radians(ego_tf.rotation.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx = loc.x - ego_tf.location.x
        dy = loc.y - ego_tf.location.y
        s =  dx * cy + dy * sy
        d = -dx * sy + dy * cy
        return s, d

    def _assign_blame_ego(self,
                          ego: “carla.Vehicle”,
                          other: “carla.Actor”,
                          world_map: “carla.Map”,
                          event_normal_impulse: “carla.Vector3D”):
        “””
        Return (ego_fault: bool, why: str).

        Compares each actor's closing-speed component along the ego→other axis.
        Ego is at fault when the impulse exceeds IMPULSE_MIN, the ego closing speed
        exceeds EGO_FAULT_CLOSE_SPEED_MIN, and ego's share of total closing speed
        exceeds EGO_FAULT_RATIO (relaxed by REAR_END_BONUS in rear-end cases).
        “””
        try:
            # Unit vector pointing from ego to the other actor.
            loc_e = ego.get_transform().location
            loc_o = other.get_transform().location
            n = self._unit_vec(loc_e, loc_o)

            # Velocity approach components.
            _, v_e = self._spd_and_vec(ego)
            if hasattr(other, “get_velocity”):
                _, v_o = self._spd_and_vec(other)
            else:
                v_o = carla.Vector3D(0.0, 0.0, 0.0)

            c_ego   = max(0.0, self._dot(v_e, n))     # Ego closing speed toward other.
            c_other = max(0.0, -self._dot(v_o, n))    # Other's closing speed toward ego (= -v_o·n).

            # Impulse magnitude.
            J = self._vec_norm(event_normal_impulse)

            # Determine if this is a rear-end scenario (target ahead and ego closes faster).
            s_rel, _ = self._ego_local_sd(ego.get_transform(), loc_o)
            rear_end_like = (s_rel > 0.0 and c_ego > c_other)

            r = c_ego / (c_ego + c_other + 1e-9)
            thr = self.EGO_FAULT_RATIO - (self.REAR_END_BONUS if rear_end_like else 0.0)

            if J >= self.IMPULSE_MIN and c_ego >= self.EGO_FAULT_CLOSE_SPEED_MIN and r >= thr:
                reason = f"ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
                return True, reason
            else:
                reason = f"non_ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
                return False, reason
        except Exception as e:
            return False, f"non_ego_fault: exception {e}"

    def collision_callback(self, event, vehicle, sensor):
        """
        Handle a collision event for vehicle.

        When the ego vehicle is involved, sets ego_collision and collision flags.
        When an NPC vehicle is involved, applies emergency braking only (not counted
        in ego metrics). The sensor is destroyed after the first trigger (one-shot).
        """
        if vehicle == self.ego_vehicle:
            # Attribute fault only when the ego vehicle is involved.
            self.collision = True
            self.ego_collision = True
            self.side_collision_count_vehicle = 1

        else:
            # Non-ego collision: apply emergency braking only (not counted in ego metrics).
            if vehicle in getattr(self, "vehicles", []):
                try:
                    idx = self.vehicles.index(vehicle)
                    controller = self.controllers[idx]
                    if controller:
                        controller.brake()
                except Exception:
                    pass

            # Apply brake directly via vehicle control.
            try:
                cur = vehicle.get_control()
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=getattr(cur, "steer", 0.0)))
            except Exception:
                pass

        # One-shot sensor: destroy after first trigger.
        try:
            sensor.stop()
            sensor.destroy()
            print(f"[INFO] Collision sensor {sensor.id} destroyed (one-shot).")
        except Exception:
            pass

        if sensor in getattr(self, "collision_sensors", []):
            try:
                self.collision_sensors.remove(sensor)
            except Exception:
                pass

    # ========== Red-light detection logic ==========

    def _detect_run_red_light(self):
        """
        Detect whether the ego vehicle ran a red light.

        Applies a grace window and checks whether the vehicle slowed sufficiently
        before the light turned green. Requires self.ego_vehicle; uses simulation
        time if self.world is available, otherwise falls back to wall-clock time.
        Returns True if a red-light violation is detected.
        """
        import math, time

        # Tunable thresholds (can be overridden externally via self._rl_* attributes).
        RED_STOP_WINDOW = getattr(self, "_rl_red_stop_window", 2.0)    # Grace period while red (s).
        STOP_SPEED_EPS = getattr(self, "_rl_stop_speed_eps", 0.2)      # Speed threshold for "stopped" (m/s).
        DECEL_DELTA_REQ = getattr(self, "_rl_decel_delta_req", 0.5)    # Minimum required speed drop before green (m/s).
        RECENT_GREEN_WINDOW = getattr(self, "_rl_recent_green_window", 3.0)  # Lookback window after red→green (s).

        def _now():
            # Prefer simulation time.
            if hasattr(self, "world") and self.world is not None:
                try:
                    return self.world.get_snapshot().timestamp.elapsed_seconds
                except Exception:
                    pass
            return time.time()

        def _speed_of(vehicle):
            v = vehicle.get_velocity()
            return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        def _reset_red_episode(_self):
            _self._rl_red_start_t = None
            _self._rl_v_at_red = None
            _self._rl_min_v_during_red = None

        # Initialize internal state on first call.
        if not hasattr(self, "_rl_last_tl_state"):
            self._rl_last_tl_state = None
        if not hasattr(self, "_rl_red_start_t"):
            self._rl_red_start_t = None
            self._rl_v_at_red = None
            self._rl_min_v_during_red = None

        # Basic visibility check.
        if not getattr(self, "ego_vehicle", None):
            return False
        tlight = self.ego_vehicle.get_traffic_light()
        if tlight is None:
            # No traffic light visible; reset the episode state.
            self._rl_last_tl_state = None
            _reset_red_episode(self)
            return False

        state = tlight.get_state()
        now = _now()
        speed = _speed_of(self.ego_vehicle)

        state_changed = (state != self._rl_last_tl_state)
        self._rl_last_tl_state = state

        # ================= Red light logic =================
        if state == carla.TrafficLightState.Red:
            if self._rl_red_start_t is None:
                # Just entered a red light.
                self._rl_red_start_t = now
                self._rl_v_at_red = speed
                self._rl_min_v_during_red = speed
            else:
                # Track the minimum speed recorded during the red phase.
                if self._rl_min_v_during_red is None:
                    self._rl_min_v_during_red = speed
                else:
                    self._rl_min_v_during_red = min(self._rl_min_v_during_red, speed)

            # Rule 1: still moving after the grace window expires → ran a red light.
            if (now - self._rl_red_start_t) >= RED_STOP_WINDOW and speed > STOP_SPEED_EPS:
                return True

            return False  # Red light active but no violation yet.

        # ================= Green light logic =================
        if state == carla.TrafficLightState.Green:
            if self._rl_red_start_t is not None:
                # Only evaluate within the short window immediately after a red phase.
                if (now - self._rl_red_start_t) <= RECENT_GREEN_WINDOW:
                    v_at_red = self._rl_v_at_red if self._rl_v_at_red is not None else speed
                    min_v = self._rl_min_v_during_red if self._rl_min_v_during_red is not None else speed
                    decel_amt = max(0.0, v_at_red - min_v)
                    slowed_enough = (decel_amt >= DECEL_DELTA_REQ) or (min_v <= STOP_SPEED_EPS)
                    if not slowed_enough:
                        _reset_red_episode(self)
                        return True
            _reset_red_episode(self)
            return False

        # ================= Yellow light logic (no violation check, state maintenance only) =================
        if state == carla.TrafficLightState.Yellow:
            # Transition from red to yellow ends the red-light episode.
            if state_changed and self._rl_red_start_t is not None:
                _reset_red_episode(self)
            return False

        # Other states (Off/Unknown): clear state to prevent stale data affecting future logic.
        _reset_red_episode(self)
        return False

    # ========== tick: step controllers and return signals ==========

    def tick(self):
        “””
        Execute one simulation step.

        1) Calls LaneKeepAndChangeController.run_step() for every NPC vehicle.
        2) Checks whether the ego vehicle ran a red light.
        3) Returns (signals_list, ego_collision, collision, ego_cross_solid_line, ego_run_red_light).
        “””
        signals_list = [None]*self.vehicle_num
        for i in range(self.vehicle_num):
            ctrl = self.controllers[i]
            if ctrl:
                control, signals = ctrl.run_step()
                self.vehicles[i].apply_control(control)
                signals_list[i] = signals
            else:
                signals_list[i] = None

        # Check for red-light violation only if not already recorded.
        if not self.ego_run_red_light:
            if self._detect_run_red_light():
                self.ego_run_red_light = True
                print(“[INFO] Ego vehicle ran a red light.”)

        return signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light

    # ========== Utility functions for the main script ==========

    def reconnect(self):
        """
        Closes the websocket connection and re-creates it so that data can be received again
        """
        self.ws.close()
        self.ws = create_connection(self.url)
        return

    def check_module_status(self, modules):
        """
        Checks if all modules in a provided list are enabled
        """
        module_status = self.get_module_status()
        for module, status in module_status.items():
            if not status and module in modules:
                log.warning("Warning: Apollo module {} is not running!!!".format(module))
                self.enable_module(module)
                time.sleep(1)

    def get_module_status(self):
        """
        Returns a dict where the key is the name of the module
        and value is a bool based on the module's current status
        """
        self.reconnect()
        data = json.loads(self.ws.recv())  # first recv => SimControlStatus
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        # In production, parse data and return real module status; returning an empty dict as a placeholder.
        return {}

    def get_controller(self, idx):
        """
        Return the controller for NPC vehicle at index idx (0-based).
        Returns None if the index is out of range.
        """
        if idx < 0 or idx >= len(self.controllers):
            print(f"[WARN] get_controller: index {idx} out of range (0–{len(self.controllers)-1}).")
            return None
        return self.controllers[idx]

    def get_vehicle_positions(self):
        """
        Return a list of CARLA Location objects for all NPC vehicles (excludes the ego vehicle).
        """
        positions = []
        for v in self.vehicles:
            loc = v.get_location()
            positions.append(loc)
        return positions

    def destroy_all(self):
        """
        Destroy all sensors, NPC vehicles, and (if owned) the ego vehicle.
        Resets all collision/fault flags for the next episode.
        """
        if self.lane_invasion_sensor_ego:
            self.lane_invasion_sensor_ego.listen(lambda event: None)

        # Allow CARLA to flush all pending callbacks before destroying.
        for _ in range(3):
            self.world.wait_for_tick()

        # Stop and destroy all collision sensors.
        for s in self.collision_sensors:
            try:
                s.stop()
                s.destroy()
            except:
                pass
        self.collision_sensors.clear()

        if self.lane_invasion_sensor_ego:
            try:
                self.lane_invasion_sensor_ego.stop()
                self.lane_invasion_sensor_ego.destroy()
            except:
                pass
            self.lane_invasion_sensor_ego = None

        for _ in range(3):
            self.world.wait_for_tick()

        # Finally, destroy all NPC vehicles.
        for v in self.vehicles:
            try:
                v.destroy()
                self.world.wait_for_tick()
            except:
                pass
        self.vehicles.clear()

        # Destroy the ego vehicle if this class owns it.
        if not self.external_ads and self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except:
                pass
            self.ego_vehicle = None

        # Reset state flags for the next episode.
        self.collision = False
        self.ego_collision = False
        self.multi_vehicle_collision_count = 0
        self.rear_collision_count_vehicle = 0
        self.side_collision_count_vehicle = 0
        self.collision_count_obj = 0
        self.ego_cross_solid_line = 0
        self.ego_run_red_light = False
        self.world.wait_for_tick()
        # self.world.tick()

    def enable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "START_MODULE", "value": module})
        )
        return

    def disable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "STOP_MODULE", "value": module})
        )
        return

    # ========== Destination setup ==========

    def set_destination(self):
        """
        Find the farthest same-direction waypoint from the ego vehicle's current lane
        via BFS and store it in self.ego_destination.
        If a WebSocket connection is open, also sends a RoutingRequest to Apollo (optional).
        """
        if not self.ego_vehicle:
            print("[ERROR] ego_vehicle not yet spawned; cannot set destination.")
            return

        # 1) Get the waypoint at the ego vehicle's current position.
        ego_loc = self.ego_vehicle.get_location()
        start_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if not start_wp:
            print("[ERROR] ego_vehicle waypoint is None; cannot set destination.")
            return

        import collections
        queue = collections.deque()
        visited = set()
        queue.append((start_wp, 0.0))
        same_direction_wps = []

        init_lane_id = start_wp.lane_id
        init_lane_sign = 1 if init_lane_id >= 0 else -1

        while queue:
            cur_wp, dist_so_far = queue.popleft()
            if cur_wp in visited:
                continue
            visited.add(cur_wp)

            same_direction_wps.append((cur_wp, dist_so_far))

            if dist_so_far > max_search_distance_for_destination:
                continue

            nxt_wps = cur_wp.next(step_dist_for_destination)
            for nxt_wp in nxt_wps:
                nxt_lane_sign = 1 if nxt_wp.lane_id >= 0 else -1
                if nxt_lane_sign == init_lane_sign:
                    dist_increment = cur_wp.transform.location.distance(nxt_wp.transform.location)
                    new_dist = dist_so_far + dist_increment
                    if new_dist <= (max_search_distance_for_destination + step_dist_for_destination):
                        queue.append((nxt_wp, new_dist))

        if not same_direction_wps:
            print("[WARNING] No same-direction waypoints found; set_destination failed.")
            return

        # Find the farthest waypoint.
        furthest_wp, furthest_dist = max(same_direction_wps, key=lambda x: x[1])
        self.ego_destination = furthest_wp.transform.location
        print(f"[INFO] set_destination: target (x={self.ego_destination.x:.2f}, y={self.ego_destination.y:.2f}), dist={furthest_dist:.1f}m")

        # If a WebSocket connection is open, send a RoutingRequest to Apollo (optional).
        apollo_data = fetch_localization_variable()
        if self.ws and apollo_data is not None and 'position' in apollo_data:
            try:
                yaw_deg = self.ego_vehicle.get_transform().rotation.yaw
                yaw_rad = math.radians(yaw_deg)

                msg = {
                    "type": "SendRoutingRequest",
                    "start": {
                        "x": apollo_data['position']['x'],
                        "y": apollo_data['position']['y'],
                        "z": apollo_data['position']['z'],
                        "heading": -yaw_rad,
                    },
                    "end": {
                        "x": self.ego_destination.x,
                        "y": -self.ego_destination.y,
                        "z": apollo_data['position']['z'],
                    },
                    "waypoint": "[]",
                }
                self.ws.send(json.dumps(msg))
                print("[INFO] Routing request sent:", json.dumps(msg))
            except WebSocketException as e:
                print(f"[ERROR] WebSocket error while sending RoutingRequest: {e}")
            except Exception as e:
                print(f"[ERROR] Internal error in set_destination: {e}")

    def close_connection(self):
        """
        Close the WebSocket connection if one is open.
        """
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
                print("[INFO] WebSocket connection closed.")
            except Exception as e:
                print(f"[ERROR] Error while closing WebSocket connection: {e}")
