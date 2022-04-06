#!/usr/bin/env python
"""Extract images from a rosbag.
"""

from __future__ import print_function, division, with_statement

import os
import errno
import argparse
import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm
import yaml
from scipy.spatial.transform import Rotation as R
import tf
from sensor_msgs.point_cloud2 import read_points

def mkdir(p):
    try:
        os.makedirs(p)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
def find_timestamps(left_lines, right_lines, lidar_lines, args):
    num_line_right = 0
    num_line_left = 0
    num_line_lidar = 0
    output_timestamps_left, output_timestamps_right, output_timestamps_lidar = [], [], []
    output_timestamps_for_atlans_camera, output_timestamps_for_atlans_lidar, output_timestamps_for_atlans_camera_lidar = [], [], []
    while num_line_right < len(right_lines):
        right_value = float(right_lines[num_line_right])
        for i in range(num_line_left, len(left_lines)):
            left_value = float(left_lines[i])
            if abs(right_value - left_value) < args.max_delay:
                break
        for j in range(num_line_lidar, len(lidar_lines)):
            lidar_value = float(lidar_lines[j])
            if abs(right_value - lidar_value) < args.max_delay:
                break
        if abs(right_value - left_value) < args.max_delay \
                and abs(right_value - lidar_value) < args.max_delay \
                and abs(left_value - lidar_value) < args.max_delay:
            output_timestamps_left.append(left_lines[i])
            output_timestamps_lidar.append(lidar_lines[j])
            output_timestamps_right.append(right_lines[num_line_right])
            camera_value = (right_value + left_value) / 2
            output_timestamps_for_atlans_camera.append(camera_value)
            output_timestamps_for_atlans_lidar.append(lidar_value)
            output_timestamps_for_atlans_camera_lidar.append((camera_value+lidar_value)/2)
            num_line_left = i
            num_line_lidar = j
            num_line_right += 1
        else:
            print('Skipping right camera timestamp:', right_lines[num_line_right].strip())
            num_line_right += 1
    return output_timestamps_left, output_timestamps_right, output_timestamps_lidar, \
        output_timestamps_for_atlans_camera, output_timestamps_for_atlans_lidar, output_timestamps_for_atlans_camera_lidar

def get_matrixes_for_rectification(config_filename):
    
    with open(config_filename, 'r') as f:
        config_cameras = yaml.load(f)
        
    fx = config_cameras['left']['intrinsics'][0]
    fy = config_cameras['left']['intrinsics'][1]
    cx = config_cameras['left']['intrinsics'][2]
    cy = config_cameras['left']['intrinsics'][3]
    matrix_intrinsics_left = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0,0,1]])
    
    fx = config_cameras['right']['intrinsics'][0]
    fy = config_cameras['right']['intrinsics'][1]
    cx = config_cameras['right']['intrinsics'][2]
    cy = config_cameras['right']['intrinsics'][3]
    matrix_intrinsics_right = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0,0,1]])
    
    distortions_left = np.array(config_cameras['left']['distortion_coeffs'])
    
    distortions_right = np.array(config_cameras['right']['distortion_coeffs'])
    
    new_matrix_intrinsics = np.array(config_cameras['left_rect']['P'])
    
    R_left = np.array(config_cameras['left']['T'])[:3,:3]
    R_left = np.linalg.inv(R_left)
    
    R_right = np.array(config_cameras['right']['T'])[:3,:3]
    R_right = np.linalg.inv(R_right)
    
    output_shape = config_cameras['left_rect']['resolution']
    
    return matrix_intrinsics_left, distortions_left, R_left, matrix_intrinsics_right, distortions_right, R_right, new_matrix_intrinsics, output_shape

def rectify_image(image, matrix_intrinsics, new_matrix_intrinsics, distortions, R, output_shape):
    height, width, channels = image.shape    
    mapx1, mapy1 = cv2.initUndistortRectifyMap(matrix_intrinsics, distortions, R, new_matrix_intrinsics,
                                               (width, height),
                                               cv2.CV_32F)
    output_image = cv2.remap(image, mapx1, mapy1, interpolation=cv2.INTER_LINEAR)
    output_image = output_image[:output_shape[1], :output_shape[0]]
    return output_image

def extract_info_from_atlans_txt(msg):
    time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    q = [
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    ]
    t = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
    return time, q, t

def matrix_from_q_t(q, t):
    matrix = tf.transformations.quaternion_matrix(q)
    matrix[:3, 3] = t
    return matrix

def save_gt_poses(bag, timestamps, outdir, args):
    gp_fname = os.path.join(outdir, 'poses.txt')
    ts_fname = os.path.join(outdir, 'timestamps.txt')
    meta_fname = os.path.join(outdir, 'meta.yaml')
    matrix_pose_0, q_0, t_0 = None, None, None
    with tqdm(total=len(timestamps)) as pbar:    
        with open(os.path.join(outdir, gp_fname), mode='w') as gp_txt:
            with open(os.path.join(outdir, ts_fname), mode='w') as ts_txt:
                index = 0
                time_nearest, matrix_pose_nearest, q_nearest, t_nearest = None, None, None, None
                delay_min = np.inf
                for topic, msg, t in bag.read_messages(topics=[args.atlans_topic]):
                    time, q, t = extract_info_from_atlans_txt(msg)
                    matrix_pose = matrix_from_q_t(q, t)
                    if index >= len(timestamps):
                        break
                    delay = abs(timestamps[index] - time)
                    if delay < delay_min:
                        delay_min = delay
                        time_nearest = time
                        matrix_pose_nearest, q_nearest, t_nearest = matrix_pose, q, t
                    if timestamps[index] < time:
                        if delay_min > args.max_delay:
                            print('No data from atlans found for timestamp', timestamps[index])
                        else:
                            if matrix_pose_0 is None:
                                matrix_pose_0, q_0, t_0 = matrix_pose_nearest, q_nearest, t_nearest

                            matrix_pose_nearest = np.dot(np.linalg.inv(matrix_pose_0), matrix_pose_nearest)
                            matrix_1D = np.reshape(matrix_pose_nearest[:3,:],(-1))
                            string_matrix = " ".join([repr(elem) for elem in matrix_1D])
                            output = string_matrix + '\n'
                            gp_txt.write(output)
                            ts_txt.write('{:.9f}\n'.format(time_nearest))

                        time_nearest, matrix_pose_nearest = None, None
                        delay_min = np.inf

                        index += 1
                        pbar.update()
    meta = {
        'atlans_initial_pose': {
            'orientation': {'x': q_0[0], 'y': q_0[1], 'z': q_0[2], 'w': q_0[3]},
            'position': {'x': t_0[0], 'y': t_0[1], 'z': t_0[2]}
        }
    }
    with open(meta_fname, 'w') as meta_f:
        yaml.dump(meta, meta_f, default_flow_style=False)

"""------------------------Example of camera config file to give as argument to the script----------------------------
left:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [1411.443857274353, 1411.443857274353, 874.8474693856422, 623.5560986832043]
  distortion_coeffs: [-0.15909806562168644, 0.09317210409700072, -0.0005030229568451252, -0.0016410631872930729, 0.0]
  resolution: [1874, 1216]
  T:
  - [0.9999639600436523, 0.0037817586269674047, -0.007601112780658122, 0.0]
  - [-0.003781649374652499, 0.9999928491252762, 2.8745779390306684e-05, 0.0]
  - [0.007601167135652063, -3.337479218771394e-17, 0.9999711107117925, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
left_rect:
  camera_model: pinhole
  intrinsics: [867.1920188938037, 867.1920188938037, 554.9548231998494, 308.58383552157835]
  resolution: [1200, 600]
  P:
  - [867.1920188938037, 0.0, 554.9548231998494, 0.0]
  - [0.0, 867.1920188938037, 308.58383552157835, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
right:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [1413.310491278634, 1413.310491278634, 876.8915066166779, 613.5510884575937]
  distortion_coeffs: [-0.16115190617719044, 0.10012927740048728, 0.0002925989395310641, -0.0009308465705273536, 0.0]
  resolution: [1874, 1216]
  T:
  - [0.9999904773933523, 8.451589067470574e-05, 0.00436325333661849, 0.0]
  - [-9.625989737719736e-05, 0.999996373459245, 0.002691431290298636, 0.0]
  - [-0.00436301004438975, -0.0026918256671755528, 0.9999868590226224, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
right_rect:
  camera_model: pinhole
  intrinsics: [867.1920188938037, 867.1920188938037, 554.9548231998494, 308.58383552157835]
  resolution: [1200, 600]
  P:
  - [867.1920188938037, 0.0, 554.9548231998494, -346.0853706022073]
  - [0.0, 867.1920188938037, 308.58383552157835, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
--------------------------------------------------------------------------------------------------------------------------"""

def main():
    argparser = argparse.ArgumentParser(description='Extract images, lidar points and odometry from a rosbag.')
    argparser.add_argument('bag_path', help='path to .bag file')
    argparser.add_argument('--config', help='path to camera configuration .yml file, see example above', required=True)
    argparser.add_argument('--left_image_topic', help='left camera image topic name (type: sensor_msgs/Image)', default = '/stereo/left/image_raw')
    argparser.add_argument('--right_image_topic', help='right camera image topic name (type: sensor_msgs/Image)', default = '/stereo/right/image_raw')
    argparser.add_argument('--lidar_topic', help='lidar topic name (type: sensor_msgs/PointCloud2)', default='/velodyne_points')
    argparser.add_argument('--atlans_topic', help='atlans odom topic name (type: nav_msgs/Odometry)', default = '/atlans_odom')
    argparser.add_argument('--encoding', help='new encoding (mono8, mono16, bgr8, rgb8, bgra8, rgba8)', default = 'bgr8')
    argparser.add_argument('--outdir', '-o', help='output dirname', default='')
    argparser.add_argument('--max_delay', '-d', type=float, help='maximum delay between two timestamps that can be considered as one time, by default it should be 1/(2*fps) (strictly < )', default=0.05)
    args = argparser.parse_args()

    print('INFO: used topic names...')
    print('....: left_image_topic:{0:>30s}'.format(args.left_image_topic))
    print('....: right_image_topic:{0:>30s}'.format(args.right_image_topic))
    print('....: lidar_topic:{0:>30s}'.format(args.lidar_topic))

    bag_path = os.path.abspath(args.bag_path)
    keeped_topics = [
        args.left_image_topic,
        args.right_image_topic
        ]
    if args.outdir != '':
        outdir = os.path.abspath(args.outdir)
    else:
        outdir = os.path.join(os.path.dirname(bag_path),
                              os.path.basename(bag_path).split('.bag')[0])
    mkdir(outdir)

    limg_dir = os.path.join(
        outdir,
        args.left_image_topic[1:].replace('/', '_')
        )
    mkdir(limg_dir)
    mkdir(os.path.join(limg_dir, 'images'))

    rimg_dir = os.path.join(
        outdir,
        args.right_image_topic[1:].replace('/', '_')
        )
    mkdir(rimg_dir)
    mkdir(os.path.join(rimg_dir, 'images'))

    lidar_dir = os.path.join(
        outdir,
        args.lidar_topic[1:].replace('/', '_')
    )
    mkdir(lidar_dir)
    mkdir(os.path.join(lidar_dir, 'clouds'))

    atlans_dir = os.path.join(
        outdir,
        args.atlans_topic[1:].replace('/', '_')
    )
    mkdir(atlans_dir)
    mkdir(os.path.join(atlans_dir, 'camera_timestamps'))
    mkdir(os.path.join(atlans_dir, 'lidar_timestamps'))
    mkdir(os.path.join(atlans_dir, 'camera_lidar_timestamps'))

    matrix_intrinsics_left, distortions_left, R_left, matrix_intrinsics_right, distortions_right, R_right, new_matrix_intrinsics, output_shape = get_matrixes_for_rectification(args.config)
    
    print('INFO: extract data to: {0}'.format(outdir))
    print('Opening bag file for reading ...')
    bridge = CvBridge()
    with rosbag.Bag(bag_path, mode='r') as bag:
        #----------------------reading timestamps to variables for left and right cameras and lidar------------------
        timestamps_lines_left_cam, timestamps_lines_right_cam, timestamps_lines_lidar = [], [], []
        print('Reading timestamps for left camera:')
        with tqdm(total=bag.get_message_count(topic_filters=[args.left_image_topic])) as pbar:
            for topic, msg, t in bag.read_messages(topics=[args.left_image_topic]):
                timestamps_lines_left_cam.append('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                pbar.update()
        pbar.close()
        print('Reading timestamps for right camera:')
        with tqdm(total=bag.get_message_count(topic_filters=[args.right_image_topic])) as pbar:
            for topic, msg, t in bag.read_messages(topics=[args.right_image_topic]):
                timestamps_lines_right_cam.append('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                pbar.update()
        pbar.close()
        print('Reading timestamps for lidar:')
        with tqdm(total=bag.get_message_count(topic_filters=[args.lidar_topic])) as pbar:
            for topic, msg, t in bag.read_messages(topics=[args.lidar_topic]):
                timestamps_lines_lidar.append('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                pbar.update()
        pbar.close()
        #--------------------end of reading timestamps----------------------------------------------------
        
        #---finding common timestamps
        output_timestamps_left, output_timestamps_right, output_timestamps_lidar, \
            timestamps_for_atlans_camera, timestamps_for_atlans_lidar, timestamps_for_atlans_camera_lidar = \
                find_timestamps(timestamps_lines_left_cam, timestamps_lines_right_cam, timestamps_lines_lidar, args)
        print('Found '+str(len(output_timestamps_left))+' common timestamps')

        print('Extracting and rectifying left and right images:')
        with tqdm(total=2*len(output_timestamps_left)) as pbar:
            with open(os.path.join(limg_dir, 'timestamps.txt'), mode='w') as ts_txt:
                with open(os.path.join(limg_dir, 'filenames.txt'), mode='w') as fn_txt:
                    index = 0
                    for topic, msg, t in bag.read_messages(topics=[args.left_image_topic]):
                        if index == 0:
                            meta = dict(frame_id=msg.header.frame_id, topic=topic)
                            with open(os.path.join(limg_dir, 'meta.yaml'), 'w') as meta_f:
                                yaml.dump(meta, meta_f, default_flow_style=False)
                        if not ('{:.9f}\n'.format(msg.header.stamp.to_sec()) in output_timestamps_left):
                            continue
                        file_name = '{0:08d}.png'.format(index)
                        ts_txt.write('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                        fn_txt.write('{0}\n'.format(file_name))

                        png_path = os.path.join(
                            limg_dir, 'images', file_name)
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=args.encoding)
                        cv_img = rectify_image(cv_img, matrix_intrinsics_left, new_matrix_intrinsics, distortions_left, R_left, output_shape)
                        cv2.imwrite(png_path, cv_img)

                        index += 1
                        pbar.update()
            with open(os.path.join(rimg_dir, 'timestamps.txt'), mode='w') as ts_txt:
                with open(os.path.join(rimg_dir, 'filenames.txt'), mode='w') as fn_txt:
                    index = 0
                    for topic, msg, t in bag.read_messages(topics=[args.right_image_topic]):
                        if index == 0:
                            meta = dict(frame_id=msg.header.frame_id, topic=topic)
                            with open(os.path.join(rimg_dir, 'meta.yaml'), 'w') as meta_f:
                                yaml.dump(meta, meta_f, default_flow_style=False)
                        if not ('{:.9f}\n'.format(msg.header.stamp.to_sec()) in output_timestamps_right):
                            continue
                        file_name = '{0:08d}.png'.format(index)
                        ts_txt.write('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                        fn_txt.write('{0}\n'.format(file_name))

                        png_path = os.path.join(
                            rimg_dir, 'images', file_name
                            )
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding=args.encoding)
                        cv_img = rectify_image(cv_img, matrix_intrinsics_right, new_matrix_intrinsics, distortions_right, R_right, output_shape)
                        cv2.imwrite(png_path, cv_img)

                        index += 1
                        pbar.update()

        print('Extracting lidar point clouds:')
        with tqdm(total=len(output_timestamps_lidar)) as pbar:
            with open(os.path.join(lidar_dir, 'timestamps.txt'), mode='w') as ts_txt:
                with open(os.path.join(lidar_dir, 'filenames.txt'), mode='w') as fn_txt:
                    index = 0
                    for topic, msg, t in bag.read_messages(topics=[args.lidar_topic]):
                        if index == 0:
                            meta = dict(frame_id=msg.header.frame_id, topic=topic)
                            with open(os.path.join(lidar_dir, 'meta.yaml'), 'w') as meta_f:
                                yaml.dump(meta, meta_f, default_flow_style=False)
                        if not ('{:.9f}\n'.format(msg.header.stamp.to_sec()) in output_timestamps_lidar):
                            continue
                        file_name = '{0:08d}.bin'.format(index)
                        ts_txt.write('{:.9f}\n'.format(msg.header.stamp.to_sec()))
                        fn_txt.write('{0}\n'.format(file_name))

                        bin_path = os.path.join(
                            lidar_dir, 'clouds', file_name)

                        points = np.array(list(read_points(msg)), dtype=np.float32)
                        points[:, 3] /= 255.  # to keep compatibility with KITTI
                        points[:, :4].tofile(bin_path)  # x, y, z, intensity

                        index += 1
                        pbar.update()

        print('Searching and saving geo poses with timestamps similiar to cameras\':')
        save_gt_poses(
            bag,
            timestamps_for_atlans_camera,
            os.path.join(atlans_dir, 'camera_timestamps'),
            args)

        print('Searching and saving geo poses with timestamps similiar to lidar\':')
        save_gt_poses(
            bag,
            timestamps_for_atlans_lidar,
            os.path.join(atlans_dir, 'lidar_timestamps'),
            args)

        print('Searching and saving geo poses with timestamps similiar to cameras\' and lidar:')
        save_gt_poses(
            bag,
            timestamps_for_atlans_camera_lidar,
            os.path.join(atlans_dir, 'camera_lidar_timestamps'),
            args)

if __name__ == '__main__':
    main()
