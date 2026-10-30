# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from data_engine_trellis.datasets.entity.door import DoorEntity


class WindowEntity(DoorEntity):
    COMMAND_STRING = "make_window"
    LANGUAGE_STRING = "window"
    CLASS_ENUM = 2000

    TOKEN = 3
