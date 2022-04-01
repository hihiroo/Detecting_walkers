# -- coding: utf-8 --
import cv2
from matplotlib.pyplot import gray
import numpy as np
import os
import sys
import argparse
import shutil

class loc(object):
    def __init__(self, info):
        self.x = info[0]
        self.y = info[1]
        self.z = info[2]
        
class rot(object):
    def __init__(self, info):
        self.yaw = info[0]
        self.roll = info[1]
        self.pitch = info[2]

def get_matrix(rotation, location):
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

class Extent(object):
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        
class Camera(object):
    def __init__(self, setting, info):
        self.location = loc(list(map(float,info[0:3])))
        self.rotation = rot(list(map(float,info[3:6])))
        self.height, self.width, self.fov = list(map(float,setting))
        
        calibration = np.identity(3)
        calibration[0,2] = self.width / 2.0
        calibration[1,2] = self.height / 2.0
        calibration[0,0] = calibration[1,1] = self.width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        self.calibration = calibration

class Walker(object):
    def __init__(self, id, info, cam):
        self.id = int(id)
        self.bb_extent = Extent(info[0], info[1], info[2])
        self.bb_location = loc(list(map(float,info[3:6])))
        self.bb_rotation = rot(list(map(float,[0,0,0])))
        self.location = loc(list(map(float,info[6:9]))) # UE4 world location x, y, z
        self.rotation = rot(list(map(float,info[9:12]))) # rotation yaw, roll, pitch
        self.cam = cam
        self.bb2d = self.get_2d_bounding_box()
        self.dist = self.get_distance()

    def get_distance(self):
        '''
            return (cam_x - walker_x)^2 + (cam_y - walker_y)^2
            not consider z axis distance
        '''
        cam_loc = self.cam.location
        obj_loc = self.location
        return (cam_loc.x-obj_loc.x)*(cam_loc.x-obj_loc.x) + (cam_loc.y-obj_loc.y)*(cam_loc.y-obj_loc.y)

    def get_bounding_box(self):
        bb_cords = self.create_bb_points()
        cords_x_y_z = self.walker_to_sensor(bb_cords)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(self.cam.calibration, cords_y_minus_z_x))
        bb = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        
        if not all(bb[:,2] > 0) and not all(bb[:,2] < 60): 
            return None
        else:
            return bb
    
    def get_2d_bounding_box(self):
        bb = self.get_bounding_box()
        if bb is None: 
            return None
        
        points = [(int(bb[i,0]), int(bb[i,1])) for i in range(8) if bb[i,0] >= 0 and bb[i,1] >= 0]
        if len(points) != 8:
            return None
        
        min_x, min_y = 100000, 100000
        max_x, max_y = 0, 0
        
        for point in points:
            if point[0] < min_x:
                min_x = point[0]
            if point[0] > max_x:
                max_x = point[0]
            if point[1] < min_y:
                min_y = point[1]
            if point[1] > max_y:
                max_y = point[1]
        
        if (max_x-min_x) < 17 or (max_y-min_y) < 35: return None
        return ((min_x, min_y), (max_x, max_y))
    
    def create_bb_points(self):
        """
        Returns 3D bounding box for walker.
        """
        cords = np.zeros((8,4))
        extent = self.bb_extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords
        
    def walker_to_sensor(self, cords):
        world_cord = self.walker_to_world(cords)
        sensor_cord = self.world_to_sensor(world_cord)
        return sensor_cord
        
    def walker_to_world(self, cords):
        bb_walker_matrix = get_matrix(self.bb_rotation, self.bb_location)
        walker_world_matrix = get_matrix(self.rotation, self.location)
        bb_world_matrix = np.dot(walker_world_matrix, bb_walker_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords
        
    def world_to_sensor(self, cords):
        sensor_world_matrix = get_matrix(self.cam.rotation, self.cam.location)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

sys.setrecursionlimit(10**6)

if __name__ == '__main__':
    '''
     Config
    '''
    argparser = argparse.ArgumentParser(
        description='bounding box maker')
    argparser.add_argument(
        '--save',
        action="store_true")
    argparser.add_argument(
        '--show',
        action="store_true")
    argparser.add_argument(
        '--debug',
        action="store_true")
    argparser.add_argument(
        '--root_path',
        default="/home/adriv/Carla/CARLA_0.9.8/PythonAPI/custom/" # 마지막에 /로 끝나게 해야함
    )
    args = argparser.parse_args()
    
    frame = 0
    dir_path = args.root_path + "image_recording/"     
    save_path = args.root_path + "gt/"
    debug_path = args.root_path + "debug/"
    
    file_list = os.listdir(dir_path)
    file_list.sort()
    
    if args.save and not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.save and not os.path.exists(save_path + 'info/'):
        os.makedirs(save_path + 'info/')
    if args.debug and not os.path.exists(debug_path):
        os.makedirs(debug_path)
    if args.save:
        gt = open(save_path + "gt.txt", 'w')
        


    for img_path in file_list:
        tmp = img_path.split('.')
        if len(tmp) == 1 or tmp[1] == 'txt': continue
        num = tmp[0]
    
        if not os.path.exists(dir_path + num + ".txt"): continue
        if not os.path.exists(dir_path + "semantic/" + num + ".png"): continue
        
        src = cv2.imread(dir_path + num + ".png", cv2.IMREAD_COLOR)
        if src is None: continue
        semantic = cv2.imread(dir_path + "semantic/" + num + ".png", cv2.IMREAD_COLOR)
        label = open(dir_path + num + ".txt", 'r')
            
        if args.save:
            gt_img = cv2.imwrite(save_path + "{0:06d}.jpg".format(frame), src)
            shutil.copyfile(dir_path + num + ".txt", save_path + "info/{0:06d}.txt".format(frame))
            
        src = cv2.putText(src, str(frame), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2) # 프레임 번호 표시
        frame += 1


        vehicle_vel = label.readline().split()
        cam_setting = label.readline().split()[1:]
        cam_info = label.readline().split()[1:]
        cam = Camera(cam_setting, cam_info)
        
        walkers = []
        while True:
            line = label.readline()
            if line is None or line == '\n': break
            line_splited = line.split()
            if(len(line_splited) < 10): break
            
            obj = Walker(line_splited[1], line_splited[2:], cam)
            walkers.append(obj)
        
        walkers.sort(key=lambda x: x.dist)
        
        h, w = semantic.shape[0:2]
        instance = [[-1 for _ in range(w)] for _ in range(h)] # h x w 빈 행렬 만들기
        
        for walker in walkers:
            bb = walker.bb2d
            if bb is None: continue
            
            (minx, miny), (maxx, maxy) = bb[0], bb[1]
            total = (maxx - minx + 1) * (maxy - miny + 1)
            cnt = 0
            
            for y in range(miny, maxy+1):
                if y >= h: break
                for x in range(minx, maxx+1):
                    if x >= w: break
                    if instance[y][x] == -1 and all(semantic[y][x] == (60,20,220)):
                        instance[y][x] = walker.id
                        cnt += 1
            
            if total > 0 and 10*cnt < 2*total: continue # cnt/total < 0.1
            cv2.rectangle(src, (minx, miny), (maxx, maxy), (0,255,0), 2) # 2차원 박스

            if args.save:
                gt.write(str(frame) + "," + str(walker.id) + "," + str(minx) + "," + str(miny) + "," + str(maxx-minx+1) + "," + str(maxy-miny+1) + "," + str(1) + "," + str(1) + "," + "1" + "\n")
            
        label.close()
        if args.show: 
            cv2.imshow("result",src)
            cv2.waitKey(10)
        if args.debug: 
            cv2.imwrite(debug_path + "{0:06d}.jpg".format(frame), src)
    if args.save:
        gt.close()
    if args.show: 
        cv2.destroyAllWindows()
