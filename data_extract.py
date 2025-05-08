import os
from new_test_data_gen import *

POSE_NAME = 'Boat_Pose'
VIDEO_NAME = '221004_yogi_nexus_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii'

if __name__ == "__main__":

    # new_train_json_file = os.path.join('./data', '4dgs/transforms_train.json')
    # old_train_json_file = os.path.join('./data', 'dnerf/transforms_train.json')

    # json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_train_json_file, new_train_json_file)


    # new_test_json_file = os.path.join('./data', '4dgs/transforms_test.json')
    # old_test_json_file = os.path.join('./data', 'dnerf/transforms_test.json')

    # json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_test_json_file, new_test_json_file)

    new_val_json_file = os.path.join('./data', '4dgs/transforms_val.json')
    old_val_json_file = os.path.join('./data', 'dnerf/transforms_val.json')

    json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_val_json_file, new_val_json_file)

    # export_to_4dgs_format(POSE_NAME, VIDEO_NAME, 50, 800, 10)
