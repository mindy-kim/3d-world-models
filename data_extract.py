import os
from new_test_data_gen import *

POSE_NAME = 'Boat_Pose'
VIDEO_NAME = '221004_yogi_nexus_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii'

if __name__ == "__main__":
    export_to_colmap_format(POSE_NAME, VIDEO_NAME)
