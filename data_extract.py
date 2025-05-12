import os
from test_data_gen import *

POSE_NAME = 'Boat_Pose'
VIDEO_NAME = '221004_yogi_nexus_body_hands_03596_Boat_Pose_or_Paripurna_Navasana_-a_stageii'

if __name__ == "__main__":

    train_path = os.path.join('./data', '4dgs/train')
    os.makedirs(train_path, exist_ok=True)
    new_train_json_file = os.path.join('./data/4dgs', 'transforms_train.json')
    old_train_json_file = os.path.join('./data', 'dnerf1/transforms_train.json')
    json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_train_json_file, new_train_json_file, 'train')


    test_path = os.path.join('./data', '4dgs/test')
    os.makedirs(test_path, exist_ok=True)
    new_test_json_file = os.path.join('./data/4dgs', 'transforms_test.json')
    old_test_json_file = os.path.join('./data', 'dnerf1/transforms_test.json')
    json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_test_json_file, new_test_json_file, 'test')



    val_path = os.path.join('./data', '4dgs/val')
    os.makedirs(val_path, exist_ok=True)
    new_val_json_file = os.path.join('./data/4dgs', 'transforms_val.json')
    old_val_json_file = os.path.join('./data', 'dnerf1/transforms_val.json')
    json_to_4dgs_format(POSE_NAME, VIDEO_NAME, old_val_json_file, new_val_json_file, 'val')

    # export_to_4dgs_format(POSE_NAME, VIDEO_NAME, 50, 800, 10)
