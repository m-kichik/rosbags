import os
import bagpy
from bagpy import bagreader

if __name__ == '__main__':
    path2bag = os.path.abspath('case30_graffiti_2022-03-22-17-06-29_0.bag')
    bag = bagreader(path2bag)
    print(bag.topic_table)
    left_cam_info_msg = bag.message_by_topic(topic='/sensum/left/image_rect_color')
    print(left_cam_info_msg)