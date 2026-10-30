import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.transform import Rotation as R
from data_engine_trellis.datasets.entity import WallEntity, DoorEntity, WindowEntity, BboxEntity

QUAD_VERTICES = (
    np.array(
        [
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
        ]
    )
    * 0.5
)


class BBoxUtils:
    def __init__(self, entities: list):
        self.entities = entities

        self.walls = []
        self.doors = []
        self.windows = []
        self.bboxes = []
        self.initialze()

    def initialze(self):
        for entity in self.entities:
            if isinstance(entity, WallEntity):
                self.walls.append(entity)
            elif isinstance(entity, DoorEntity):
                self.doors.append(entity)
            elif isinstance(entity, WindowEntity):
                self.windows.append(entity)
            elif isinstance(entity, BboxEntity):
                self.bboxes.append(entity)
            else:
                raise ValueError(f"Unknown entity type: {type(entity)}")

    def extract_walls(self):
        all_walls = {}
        room_walls = []
        room_wall_start = 0
        room_count = 0
        for wi, wall in enumerate(self.walls):
            wall_start = self.walls[room_wall_start]
            ax = wall_start.params["a_x"]
            ay = wall_start.params["a_y"]
            bx = wall.params["b_x"]
            by = wall.params["b_y"]
            start_pt = np.array([ax, ay])
            end_pt = np.array([bx, by])
            dist = np.linalg.norm(start_pt - end_pt)
            if dist < 1e-4:
                room_walls.append(wall)
                all_walls[room_count] = room_walls
                room_count += 1
                room_walls = []
                room_wall_start = wi + 1
            else:
                room_walls.append(wall)
        return all_walls

    def extract_dw(self):
        # extract door, window entities from object bboxes
        doors = []
        windows = []
        for bbox in self.bboxes:
            if bbox.params["class"] == "door":
                doors.append(bbox)
            elif bbox.params["class"] == "window":
                windows.append(bbox)
            else:
                continue

        new_door_entities = self._extract_fixture(doors, self.walls, DoorEntity)
        new_window_entities = self._extract_fixture(windows, self.walls, WindowEntity)
        self.doors += new_door_entities
        self.windows += new_window_entities
        return self.doors, self.windows

    @staticmethod
    def _extract_fixture(fixture_bboxes, walls, entity_class=DoorEntity):
        new_entities = []
        for fi, fixture in enumerate(fixture_bboxes):
            fid = fixture.params["id"]
            px = fixture.params["position_x"]
            py = fixture.params["position_y"]
            pz = fixture.params["position_z"]
            z_angle = fixture.params["angle_z"]
            scale_x = np.abs(fixture.params["scale_x"])
            scale_y = np.abs(fixture.params["scale_y"])
            scale_z = np.abs(fixture.params["scale_z"])

            translate_vec = np.array([px, py, pz])
            rot_2d = R.from_rotvec(np.array([0, 0, 1]) * z_angle).as_matrix()
            scale_mat = np.diag([scale_x, scale_y, scale_z])

            fixture_xyz = rot_2d @ scale_mat @ QUAD_VERTICES.T + translate_vec.reshape(
                3, 1
            )
            fixture_xy = fixture_xyz[:2].T
            fixture_center = np.mean(fixture_xy, axis=0)
            fixture_direction = fixture_xy[2] - fixture_xy[0]
            fixture_direction = fixture_direction / np.linalg.norm(fixture_direction)

            min_wall_dist = np.inf
            best_wall_data = None
            for wi, wall in enumerate(walls):
                # find the wall that is closest to the door
                wid = int(wall.params["id"])
                ax = wall.params["a_x"]
                ay = wall.params["a_y"]
                bx = wall.params["b_x"]
                by = wall.params["b_y"]
                wpa = np.array([ax, ay])
                wpb = np.array([bx, by])
                wall_direction = wpb - wpa
                wall_length = np.linalg.norm(wall_direction)
                wall_direction = wall_direction / wall_length

                # check if the door is parallel to the wall
                if np.abs(np.dot(wall_direction, fixture_direction)) < 0.5:
                    continue

                # project the center of the door on the wall
                pt = fixture_center - wpa
                projected_length = np.dot(pt, wall_direction)
                if projected_length <= 0 or projected_length >= wall_length:
                    continue

                # point to line distance
                door_wall_dist = np.abs(np.cross(wall_direction, fixture_center - wpa))
                if door_wall_dist < min_wall_dist:
                    min_wall_dist = door_wall_dist
                    best_wall_data = [wi, door_wall_dist, projected_length]

            if best_wall_data is None:
                continue

            best_wall_idx, best_wall_dist, projected_length = best_wall_data
            best_wall = walls[best_wall_idx]

            if best_wall_dist > 2 * scale_y:
                continue

            wall_start = np.array([best_wall.params["a_x"], best_wall.params["a_y"]])
            wall_end = np.array([best_wall.params["b_x"], best_wall.params["b_y"]])
            wall_direction = wall_end - wall_start
            wall_direction = wall_direction / np.linalg.norm(wall_direction)

            center_pt = projected_length * wall_direction + wall_start

            door_entity = entity_class(
                {
                    "id": fid,
                    "wall0_id": best_wall.params["id"],
                    "wall1_id": best_wall.params["id"],
                    "position_x": center_pt[0],
                    "position_y": center_pt[1],
                    "position_z": pz,
                    "width": scale_x,
                    "height": scale_z,
                }
            )
            new_entities.append(door_entity)

        return new_entities

    def extract_objects(self):
        new_entities = []
        for bbox in self.bboxes:
            if bbox.params["class"] in ["floor", "ceiling", "door", "window"]:
                continue
            elif bbox.params["class"] == "wall":
                # check the thickness of the wall
                scale_x = np.abs(bbox.params["scale_x"])
                scale_y = np.abs(bbox.params["scale_y"])
                scale_z = np.abs(bbox.params["scale_z"])
                if scale_x < 0.05 or scale_y < 0.05 or scale_z < 0.05:
                    continue

                bbox.params["class"] = "wall_cabinet"
            elif bbox.params["class"] == "unknown":
                bbox.params["class"] = "other"

            new_entities.append(bbox)
        self.bboxes = new_entities
        return self.bboxes

    def inside_wall(self):
        wall_points = []
        for wall in self.walls:
            ax = wall.params["a_x"]
            ay = wall.params["a_y"]
            wall_points.append((ax, ay))

        assert (self.walls[-1].params["b_x"] == self.walls[0].params["a_x"]) and (
            self.walls[-1].params["b_y"] == self.walls[0].params["a_y"]
        )
        wall_points += [wall_points[0]]
        wall_polygon = Polygon(wall_points)

        bboxes = []
        for bbox in self.bboxes:
            px = bbox.params["position_x"]
            py = bbox.params["position_y"]
            point = Point(px, py)
            if not wall_polygon.contains(point):
                continue
            bboxes.append(bbox)
        self.bboxes = bboxes
        return self.bboxes