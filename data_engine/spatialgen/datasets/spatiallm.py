import json

import numpy as np

from utils.typing import *
from data_engine_trellis.datasets.entity.base_entity import BaseEntity
from data_engine_trellis.datasets.entity import WallEntity, DoorEntity, WindowEntity, BboxEntity

from data_engine_trellis.datasets.language_sequence import LanguageSequence


def load_layout(input: Union[str, dict]):
    """
        Load layout from a file or a dict.
    Args:
        input: Either a file path (str) or entities dict.

    Returns:
        LanguageSequence
    """

    if isinstance(input, str):
        with open(input, "r") as f:
            entities = json.load(f)
    elif isinstance(input, dict):
        entities = input
    else:
        raise ValueError("Input must be either a file path (str) or entities dict")

    wall_entities = []
    door_entities = []
    window_entities = []
    object_entities = []
    for k, entity in entities.items():
        if k == "wall":
            wall_entities = [WallEntity(params) for params in entity]
        elif k == "door":
            door_entities = [DoorEntity(params) for params in entity]
        elif k == "window":
            window_entities = [WindowEntity(params) for params in entity]
        elif k == "bbox":
            object_entities = [BboxEntity(params) for params in entity]
        else:
            raise ValueError(f"Unknown entity type: {k}")

    all_entities = wall_entities + door_entities + window_entities + object_entities
    lang_seq = LanguageSequence(all_entities)
    lang_seq.sort_entities("id")
    lang_seq.assign_doors_windows_to_walls()

    return lang_seq


def compute_layout_center_and_scale(lang_seq: LanguageSequence):
    """
        Compute the center and scale of the layout.
    Args:
        lang_seq: LanguageSequence

    Returns:
        center: np.ndarray
        scale: float
    """
    extent = lang_seq.extent()
    xyz_min = np.array([extent["min_x"], extent["min_y"], extent["min_z"]])
    xyz_max = np.array([extent["max_x"], extent["max_y"], extent["max_z"]])

    center = (xyz_min + xyz_max) / 2
    size = [extent["size_x"], extent["size_y"], extent["size_z"]]
    scale = max(size)
    return center, scale


def langseq_to_entities(langseq: LanguageSequence):

    wall = []
    door = []
    window = []
    bbox = []

    for entity in langseq.entities:
        entity: BaseEntity
        if entity.COMMAND_STRING == "make_wall":
            entity = {
                "id": entity.params["id"],
                "a_x": entity.params["a_x"],
                "a_y": entity.params["a_y"],
                "a_z": entity.params["a_z"],
                "b_x": entity.params["b_x"],
                "b_y": entity.params["b_y"],
                "b_z": entity.params["b_z"],
                "height": entity.params["height"],
                "thickness": entity.params["thickness"],
            }
            wall.append(entity)

        elif entity.COMMAND_STRING == "make_door":
            entity = {
                "id": entity.params["id"],
                "wall0_id": entity.params["wall0_id"],
                "wall1_id": entity.params["wall1_id"],
                "position_x": entity.params["position_x"],
                "position_y": entity.params["position_y"],
                "position_z": entity.params["position_z"],
                "width": entity.params["width"],
                "height": entity.params["height"],
            }
            door.append(entity)
        elif entity.COMMAND_STRING == "make_window":

            entity = {
                "id": entity.params["id"],
                "wall0_id": entity.params["wall0_id"],
                "wall1_id": entity.params["wall1_id"],
                "position_x": entity.params["position_x"],
                "position_y": entity.params["position_y"],
                "position_z": entity.params["position_z"],
                "width": entity.params["width"],
                "height": entity.params["height"],
            }

            window.append(entity)
        elif entity.COMMAND_STRING == "make_bbox":

            entity = {
                "id": entity.params["id"],
                "class": entity.params["class"],
                "position_x": entity.params["position_x"],
                "position_y": entity.params["position_y"],
                "position_z": entity.params["position_z"],
                "angle_z": entity.params["angle_z"],
                "scale_x": entity.params["scale_x"],
                "scale_y": entity.params["scale_y"],
                "scale_z": entity.params["scale_z"],
            }

            # Normalize angle to [-pi, pi] range
            angle_z = entity["angle_z"]
            angle_z = np.deg2rad(np.round(np.rad2deg(angle_z)))
            angle_z = np.arctan2(np.sin(angle_z), np.cos(angle_z))
            entity["angle_z"] = angle_z

            bbox.append(entity)
        else:
            print(f"Entity to box conversion not implemented for: cmd={entity.COMMAND_STRING}")
            continue

    return {
        "wall": wall,
        "door": door,
        "window": window,
        "bbox": bbox,
    }
