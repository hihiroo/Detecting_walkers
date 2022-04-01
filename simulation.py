#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function
from distutils.command.config import config
from distutils.spawn import spawn


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

# 에이전트 관련 import
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
# from agents.navigation.custom_agent import BasicAgent

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_k
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- Scenario ---------------------------------------------------------------------
# ==============================================================================

# carla.World + 맵 이름 입력
class Scenario(object):
    def __init__(self, carla_world, carla_map):
        # carla
        self.world = carla_world # carla.World Object
        self.map = carla_map # map name

        # runtime variables
        self.is_running = False
        self.controller_running = False

        # scenario number
        self.scenario_index = -1
        self.scenario_len = 0

        # carla.Transform list
        self.player_transforms = []     # [player_spawn_transform, map_name]
        self.car_transforms = []        # [[car1_spawn_transform, car2_spawn_transform, ...], map_name]
        self.walker_transforms = []     # [[walker1_spawn_transform, walker2_spawn_transform, ...], map_name]

        # Destination carla.Location list
        self.player_destination = []    # [player_destination_transform, map_name]
        self.car_destination = []       # [[car1_destination_transform, car2_destination_transform, ...], map_name]
        self.walker_destination = []    # [[walker1_destination_transform, walker2_destination_transform, ...], map_name]

        # Speed
        self.player_target_speed = []
        self.car_target_speed = []
        self.walker_target_speed = []

        # Current Scenario actors
        self.player = None
        self.cars = []
        self.walkers = []
        self.props = []

        # Current Scenario Controller
        self.player_agent = None
        self.car_agents = []
        self.walker_controller = []

        # Logging
        self.logger = None

        # Logger setup
        self.setup_logger()
        # self.start_logging() # 로그 출력이 불필요하다면 주석처리

        # Object setup
        self.setup_data()
                

    def toggle_running(self):
        self.is_running = not self.is_running
        self.logger.info("toggle_running() : is_running : "+str(self.is_running))


    # Python Logger 초기화
    def setup_logger(self):
        if self.logger is None:
            self.logger = logging.getLogger("Scenario")
            self.logger.setLevel(logging.WARNING)
            formatter = logging.Formatter(fmt='\n[%(levelname)s|%(name)s|%(lineno)s|%(asctime)s] %(message)s', datefmt="%H:%M:%S")
            self.logger.propagate = False
            streamHandler = logging.StreamHandler()
            streamHandler.setLevel(logging.INFO)
            streamHandler.setFormatter(formatter)
            self.logger.addHandler(streamHandler)
        else:
            self.logger.info("setup_logger() : logger already exists.")


    # Logging level INFO로 올려서 로그 활성화
    def start_logging(self):
        self.logger.setLevel(logging.INFO)


    # 시나리오 기본 데이터 초기화
    def setup_data(self):
        # Setup Actor's carla.Transfrom, carla.location data
        self.player_transforms, self.player_destination, self.player_target_speed = self.init_player()
        self.car_transforms, self.car_destination, self.car_target_speed = self.init_car()
        self.walker_transforms, self.walker_destination, self.walker_target_speed = self.init_walker()

        # 시나리오 개수가 일치하면 Valid 한 것으로 판단
        if len(self.player_transforms) == len(self.car_transforms) == len(self.walker_transforms):
            self.scenario_len = len(self.player_transforms)
        else:
            self.scenario_len = 0

        # Setup Car-Speed

        # Setup Walker-Speed

        self.logger.info("setup_data() finished with " + str(self.scenario_len) + " scenario(s).")

    
    # 시나리오 불러오기
    def call_scenario(self, scenario_no):
        if self.scenario_len and scenario_no < self.scenario_len:
            self.scenario_index = scenario_no
            self.spawn_player(scenario_no)
            self.spawn_car(scenario_no)
            self.spawn_walker(scenario_no)
            self.logger.info("call_scenario() finished.")
        else:
            self.logger.warning('call_scenario() error : Out Of Range')


    # 시나리오 진행 (루프마다 호출할 것)
    def run_scenario(self):
        # 플레이어 이동
        if self.player is not None and self.player_agent is not None:
            self.apply_control(self.player, self.player_agent)

        # 신호등 관리
        # print(self.player.is_at_traffic_light())

        # 차량 이동
        car_num = len(self.car_agents) 
        for i in range(car_num):
            car = self.cars[i]
            car_agent = self.car_agents[i]
            if car is not None and car_agent is not None:
                self.apply_control(car, car_agent)
        
        if not self.controller_running:
            self.controller_running = True
            walker_num = len(self.walker_controller)
            for i in range(walker_num):
                walker_controller = self.walker_controller[i][0]
                walker_destination = self.walker_controller[i][1]
                walker_speed = self.walker_target_speed[self.scenario_index][0][i]
                print(walker_speed)
                # 컨트롤러가 존재한다면 이동
                if walker_controller is not None:
                    walker_controller.start()
                    walker_controller.go_to_location(walker_destination)
                    walker_controller.set_max_speed(speed=walker_speed)

            self.logger.info("run_scenario() : walker controller switch on.")
        
        self.logger.info("run_scenario() called.")

    
    # actor에게 agent의 명령을 1회 실행
    def apply_control(self, actor, agent):
        control = agent.run_step()
        control.manual_gear_shift = False
        actor.apply_control(control)


    # 플레이어 반환
    def get_player(self):
        return self.player


    # 플레이어(카메라) Spawn Transform과 Destination location 초기화
    def init_player(self):
        transform_list = []
        location_list = []
        target_speed = []

        # 다음에 [스폰 위치, 맵 이름] + [도착 위치, 맵 이름] 를 추가
        # 도착위치가 None일 경우 움직이지 않는다.
        # 출발지 Transform -> 도착지 Location -> 목표 속도 순서로 작성
        transform_list.append([self.transform(258.6, -180.0, 2.0, 0.000000,270.000000,0.000000), '/Game/Carla/Maps/Town04']) # town04-0
        location_list.append([self.location(258.6, -273.2, 2.0), '/Game/Carla/Maps/Town04'])
        target_speed.append([60, '/Game/Carla/Maps/Town04'])

        transform_list.append([self.transform(-192.0, 34.0, 2.0, 0.000000,90.000000,0.000000), '/Game/Carla/Maps/Town05']) # town05-0
        location_list.append([self.location(-182.4, 140.0, 2.0), '/Game/Carla/Maps/Town05'])
        target_speed.append([60, '/Game/Carla/Maps/Town05'])

        # 맵 이름에 따라 필터링
        transform_list = filter(lambda trans: trans[1] == self.map, transform_list)
        location_list = filter(lambda loc: loc[1] == self.map, location_list)
        target_speed = filter(lambda ts: ts[1] == self.map, target_speed)

        # 만약 길이가 다른 경우 중단
        if len(transform_list) != len(location_list):
            self.logger.info("init_player() result is not valid.")
            return []

        # 로깅
        self.logger.info("init_player() results =====")
        for i in range(len(transform_list)):
            map_name = "index : " + str(i) + ", map_name : " + transform_list[i][1] + "\n"

            spawn_info = ""
            if transform_list[i][0] is None:
                spawn_info = "Object not spawned\n"
            else:
                spawn_info = "Spawned at " + str(transform_list[i][0]) + "\n"

            dest_info = ""
            if location_list[i][0] is None:
                dest_info = "Destination not set"
            else:
                dest_info = "Destination at " + str(location_list[i][0]) + "\n"
            
            map_name = map_name + spawn_info + dest_info
            self.logger.info(map_name)
        
        return transform_list, location_list, target_speed

    
    # 플레이어 스폰을 시도하고, 성공 여부를 리턴
    def spawn_player(self, scenario_no = 0):
        # 리턴값
        is_success = False

        # 설정 변수
        player_bp_id = 'vehicle.audi.a2'
        player_bp_rolename = 'hero'
        player_color = '66,66,66' # 'R,G,B' 형식으로 입력

        # 플레이어 스폰
        if not self.scenario_len:
            self.logger.warning("spawn_player() error : Needs at least 1 Scenario list")
        elif self.scenario_len <= scenario_no:
            self.logger.warning("spawn_player() error : scenario_no is Out Of Range")
        else:
            if self.player is not None: # 이미 시나리오가 진행된 경우 초기화
                self.clear_scenario('player')
            
            # 설정 변수에 따라 플레이어 Blueprint 생성
            blueprint = self.world.get_blueprint_library().find(player_bp_id)
            blueprint.set_attribute('role_name', player_bp_rolename)

            # 차량 속성에 따른 추가 설정
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', player_color)

            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'true')

            # if blueprint.has_attribute('speed'):
            #     self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            #     self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

            # 플레이어 액터 생성
            while self.player is None:
                spawn_point = self.player_transforms[scenario_no][0] if self.player_transforms[scenario_no][0] is not None else carla.Transform()
                self.player = self.world.try_spawn_actor(blueprint, spawn_point)
                if self.player is not None:
                    destination_point = self.player_destination[scenario_no][0] # 도착지점

                    if destination_point is None:
                        self.player_agent = None
                    else:
                        player_speed = self.player_target_speed[0][0]
                        self.player_agent = BasicAgent(self.player, player_speed) # 플레이어 에이전트 생성, 목표 속도 40
                        self.player_agent = self.setup_agent(self.player_agent, destination = destination_point) # 플레이어 에이전트 설정              
        
        if self.player:
            is_success = True
        
        self.logger.info('spawn_player() is ' + 'success' if is_success else 'failed')
        if is_success:
            self.logger.info("spawn_player() : player actor"+str(self.player))

        return is_success


    # 차량 스폰 위치를 초기화
    def init_car(self):
        transform_list = []
        location_list = []
        target_speed = []

        # 다음에 [[스폰 위치1, 스폰 위치2, ...], 맵 이름] + [[도착 위치1, 도착 위치2, ...], 맵 이름] 을 추가
        # 도착위치가 None일 경우 움직이지 않는다.
        # 출발지 Transform -> 도착지 Location -> 목표 속도 순서로 작성
        transform_list.append([[self.transform(269.3, -249.9, 2.0, 0.000000, 180.000000, 0.000000), ], '/Game/Carla/Maps/Town04']) # town04-0
        # location_list.append([[self.location(228.4, -249.9, 2.0), ], '/Game/Carla/Maps/Town04'])
        location_list.append([[None, ], '/Game/Carla/Maps/Town04']) # 차량 고정
        target_speed.append([[0, ], '/Game/Carla/Maps/Town04'])

        transform_list.append([[self.transform(-207.3, 94.9, 2.0, 0.000000, 0.000000, 0.000000), ], '/Game/Carla/Maps/Town05']) # town05-0
        location_list.append([[None, ], '/Game/Carla/Maps/Town05'])
        target_speed.append([[0, ], '/Game/Carla/Maps/Town05'])

        # 맵 이름에 따라 필터링
        transform_list = filter(lambda trans: trans[1] == self.map, transform_list)
        location_list = filter(lambda loc: loc[1] == self.map, location_list)
        target_speed = filter(lambda ts: ts[1] == self.map, target_speed)

        # 만약 길이가 다른 경우 중단
        if len(transform_list) != len(location_list):
            self.logger.info("init_car() result is not valid.")
            return []

        # 로깅
        self.logger.info("init_car() results =====")
        for i in range(len(transform_list)):
            map_name = "index : " + str(i) + ", map_name : " + transform_list[i][1] + "\n"
            for j in range(len(transform_list[i][0])):
                map_name = map_name + str(j) + " th car information\n"

                spawn_info = ""
                if transform_list[i][0][j] is None:
                    spawn_info = "Object not spawned\n"
                else:
                    spawn_info = "Spawned at " + str(transform_list[i][0][j]) + "\n"

                dest_info = ""
                if location_list[i][0][j] is None:
                    dest_info = "Destination not set"
                else:
                    dest_info = "Destination at " + str(location_list[i][0][j]) + "\n"
                
                map_name = map_name + spawn_info + dest_info
                self.logger.info(map_name)

        return transform_list, location_list, target_speed

    
    # 시나리오 차량 스폰을 시도하고 성공 여부를 리턴
    def spawn_car(self, scenario_no = 0):
        # 리턴값
        is_success = False

        # 설정 변수
        car_bp_id = 'vehicle.*'
        car_bp_rolename = 'obstacle_car'

        # 차량 스폰
        if not self.scenario_len:
            self.logger.warning("spawn_car() error : Needs at least 1 Scenario list")
        elif self.scenario_len <= scenario_no:
            self.logger.warning("spawn_car() error : scenario_no is Out Of Range")
        else:
            if len(self.cars) == 0: # 이미 시나리오가 진행된 경우 초기화
                self.clear_scenario('car')
            
            # 설정 변수에 따라 차량 Blueprint 생성 + 필터링
            blueprints = self.world.get_blueprint_library().filter(car_bp_id)
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            
            # 요청한 시나리오에서의 차량 정보 로드
            car_transforms = self.car_transforms[scenario_no][0]
            car_destinations = self.car_destination[scenario_no][0]
            car_target_speeds = self.car_target_speed[scenario_no][0]
            car_cnt = len(car_transforms)

            ######## 수정해야할부분? ########
            # 차량을 항상 같은 차량으로 고정
            # 만약 바꿀거면 이거 없애고 랜덤으로 bp 돌리던가 해야함
            temp = 0
            ######## ########

            # 각 차량의 스폰위치
            for idx in range(car_cnt):
                car_transform = car_transforms[idx] # 차량의 스폰위치
                car_speed = car_target_speeds[idx]
                blueprint = blueprints[temp]

                blueprint.set_attribute('role_name', car_bp_rolename)

                # 차량 속성에 따른 추가 설정
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)

                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                
                car = self.world.try_spawn_actor(blueprint, car_transform)
                self.logger.info('spawn_car()-'+str(idx)+' : car actor '+str(car))
                self.cars.append(car)

                # 차량 에이전트 생성
                if car is not None:
                    destination_point = car_destinations[idx] # 차량의 도착위치
                    print(destination_point)

                    if destination_point is None:
                        car_agent = None
                    else:
                        # 차량 에이전트 설정
                        car_agent = BasicAgent(car, car_speed)
                        car_agent = self.setup_agent(car_agent, destination = destination_point)
                    
                    self.car_agents.append(car_agent)


        if self.cars:
            is_success = True

        self.logger.info('spawn_car() is ' + 'success' if is_success else 'failed')

        return is_success


    # 보행자 스폰 위치 초기화
    def init_walker(self):
        transform_list = []
        location_list = []
        target_speed = []

        # 다음에 [[스폰 위치1, 스폰 위치2, ...], 맵 이름] + [[도착 위치1, 도착 위치2, ...], 맵 이름] 을 추가
        # 도착위치가 None일 경우 움직이지 않는다.
        transform_list.append([[self.transform(277.0, -253.7, 2.0, 0.000000, 180.000000, 0.000000), ], '/Game/Carla/Maps/Town04']) # town04-0
        location_list.append([[self.location(235.0, -253.7, 2.0), ], '/Game/Carla/Maps/Town04'])
        target_speed.append([[6.0, ], '/Game/Carla/Maps/Town04'])

        transform_list.append([[self.transform(-223.6, 98.4, 2.0, 0.000000, 0.000000, 0.000000), ], '/Game/Carla/Maps/Town05']) # town05-0
        location_list.append([[self.location(-171.0, 98.4, 2.0), ], '/Game/Carla/Maps/Town05'])
        target_speed.append([[6.0, ], '/Game/Carla/Maps/Town05'])

        # 만약 길이가 다른 경우 중단
        if len(transform_list) != len(location_list):
            self.logger.info("init_walker() result is not valid.")
            return []

        # 맵 이름에 따라 필터링
        transform_list = filter(lambda trans: trans[1] == self.map, transform_list)
        location_list = filter(lambda loc: loc[1] == self.map, location_list)
        target_speed = filter(lambda ts: ts[1] == self.map, target_speed)

        # 로깅
        self.logger.info("init_walker() results =====")
        for i in range(len(transform_list)):
            map_name = "index : " + str(i) + ", map_name : " + transform_list[i][1] + "\n"
            for j in range(len(transform_list[i][0])):
                map_name = map_name + str(j) + " th walker information\n"

                spawn_info = ""
                if transform_list[i][0][j] is None:
                    spawn_info = "Object not spawned\n"
                else:
                    spawn_info = "Spawned at " + str(transform_list[i][0][j]) + "\n"

                dest_info = ""
                if location_list[i][0][j] is None:
                    dest_info = "Destination not set"
                else:
                    dest_info = "Destination at " + str(location_list[i][0][j]) + "\n"
                
                map_name = map_name + spawn_info + dest_info
                self.logger.info(map_name)

        return transform_list, location_list, target_speed


    # 보행자 스폰
    def spawn_walker(self, scenario_no = 0):
        # 리턴값
        is_success = False

        # 설정 변수
        walker_bp_id = 'walker.pedestrian.*'
        walker_bp_rolename = 'obstacle_walker'

        if not self.scenario_len:
            self.logger.warning("spawn_walker() error : Needs at least 1 Scenario list")
        elif self.scenario_len <= scenario_no:
            self.logger.warning("spawn_walker() error : scenario_no is Out Of Range")
        else:
            if len(self.cars) == 0: # 이미 시나리오가 진행된 경우 초기화
                self.clear_scenario('walker')

            # 설정 변수에 따라 보행자 Blueprint 생성 + 필터링
            walker_blueprints = self.world.get_blueprint_library().filter(walker_bp_id)

            # 요청한 시나리오에서의 차량 정보 로드
            walker_transforms = self.walker_transforms[scenario_no][0]
            walker_destinations = self.walker_destination[scenario_no][0]
            walker_cnt = len(walker_transforms)

            # 보행자 속도 저장 리스트
            walker_speeds = self.walker_target_speed[scenario_no]

            temp = 0 # 수정해야할부분 -> 보행자 고정 하는 변수임

            for idx in range(walker_cnt):
                walker_transform = walker_transforms[idx] # 보행자 스폰위치
                blueprint = walker_blueprints[temp]
                walker_speed = walker_speeds[0][idx]
                # print(walker_speed)

                blueprint.set_attribute('role_name', walker_bp_rolename)

                # 보행자 속성에 따른 추가 설정
                if blueprint.has_attribute('is_invincible'):
                    blueprint.set_attribute('is_invincible', 'false')

                if blueprint.has_attribute('speed'):
                    walker_speeds.append(walker_speed) # 불러온 속도 적용
                else:
                    walker_speeds.append(0.0)
                
                # 보행자 스폰
                walker = self.world.try_spawn_actor(blueprint, walker_transform)
                               
                self.logger.info('spawn_walker()-'+str(idx)+' : walker actor '+str(walker))
                self.walkers.append(walker)

                # 보행자 컨트롤러 생성
                if walker is not None:
                    destination_point = walker_destinations[idx]
                    walker_controller = None

                    # 보행자 컨트롤러와 보행자 연결
                    if destination_point is not None:
                        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                        walker_controller = self.world.try_spawn_actor(blueprint=walker_controller_bp, transform=carla.Transform(), attach_to=walker)
                    
                    # 컨트롤러 추가
                    self.walker_controller.append([walker_controller, destination_point])

        return is_success


    # Carla Transform Object 반환
    def transform(self, x, y, z, pitch, yaw, roll):
        return carla.Transform(self.location(x, y, z), self.rotation(pitch, yaw, roll))


    # Carla location Object 반환
    def location(self, _x, _y, _z):
        return carla.Location(x = _x, y = _y, z = _z)


    # Carla Rotation Object 반환
    def rotation(self, _pitch, _yaw, _roll):
        return carla.Rotation(pitch = _pitch, yaw = _yaw, roll = _roll)

    
    # Basic Agent 기본 설정
    # 목적지가 None이면 따로 에이전트를 두지 않는다.
    def setup_agent(self, agent, destination):
        if agent is None:
            self.logger.warning("setup_agent() error : Agent not exists.")
        elif destination is None:
            self.logger.info("setup_agent() : destination not exists.")
            agent = None
        else:
            agent.set_destination((destination.x, destination.y, destination.z))
            
            ############# 테스트해야함 #################
            # 최신 버전에서 가능 -> 변경하고 켜기
            # agent.set_target_speed(50) # 속도 조절
            # agent.ignore_traffic_lights(True) # 신호등 무시
            # agent.ignore_stop_signs(True) # 정시 신호 무시
            # agent.ignore_vehicles(True) # 차량 무시
            ############# 테스트해야함 #################
            self.logger.info("setup_agent() finished.")
        return agent


    # 시나리오 종료
    ############# 테스트해야함 #################
    def clear_scenario(self, specify='All'):
        try:
            if specify in ['All', 'player']:
                # 플레이어 제거
                if self.player is not None:
                    self.player.destroy()
                    self.player = None

                # 플레이어 에이전트 제거
                # self.player_agent.destroy()
            
            if specify in ['All', 'car']:
                # 차량 제거
                for car in self.cars:
                    if car is not None:
                        car.destroy()
                    self.cars = []
                
                # 차량 에이전트 제거
                # for car_agent in self.car_agents:
                #     if car_agent is not None:
                #         car_agent.destroy()
                #     self.car_agents = []
            
            if specify in ['All', 'walker']:
                # 보행자 제거
                for walker in self.walkers:
                    if walker is not None:
                        walker.destroy()
                    self.walkers = []
                
                # 보행자 컨트롤러 제거
                for walker_controller in self.walker_controller:
                    if walker_controller is not None:
                        walker_controller.destroy()
                    self.walker_controller = []

            if specify in ['All', 'prop']:
                # 장애물 제거
                for prop in self.props:
                    if prop is not None:
                        prop.destroy()
                    self.props = []
            
            self.logger.info("clear_scenario("+specify+") successes")

        except:
            self.logger.info("clear_scenario() error.")




# WORLD #
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.scenario = Scenario(carla_world, '/Game/Carla/Maps/Town04')
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0


    def restart(self):
        ########################## 시나리오 ##########################
        scenario_no = 0
        self.scenario.call_scenario(scenario_no)

        self.player = self.scenario.get_player()

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        print('setup scenario finished.')

        ########################## 시나리오 ####################
        # 
        # 
        # ######

        # player setting


        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
    
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    # def next_prop(self):
    #     if self._props:
    #         for p in self._props:
    #             p.destroy()
    #         self._props = []
    #         print("previous prop destroyed.")
        
    #     self._prop_idx = (self._prop_idx + 1) % len(self.props_bp)
    #     prop_bp = self.props_bp[self._prop_idx]
    #     prop_bp.set_attribute('role_name', 'prop_test')
    #     self._props.append(self.world.spawn_actor(prop_bp, self._trans))
    #     return str(prop_bp.id[12:]) # name

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,]
            #self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        
        # 시나리오 관련 액터 초기화
        self.scenario.clear_scenario()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_k:
                    world.scenario.toggle_running()
                    
                    
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= carla.VehicleLightState.All ^ carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= carla.VehicleLightState.All ^ carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.file_name = None
        self.capture = False
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # if self.capture:
        #     image.save_to_disk('_out/' + self.file_name + '.jpg')
        #     print("save as" + self.file_name)
        #     self.capture = False
        
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            # image.save_to_disk('_out/%08d' % image.frame)
            image.save_to_disk('_out/' + self.file_name + '.jpg')
            self.recording = False
            print("save as " + '_out/' + self.file_name + '.jpg' )


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    world = None

    play = True
    isdebug = False

    if play:
        pygame.init()
        pygame.font.init()
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(5.0)

            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            world_temp = client.load_world('/Game/Carla/Maps/Town04')

            hud = HUD(args.width, args.height)
            world = World(world_temp, hud, args)
            controller = KeyboardControl(world, args.autopilot)

            clock = pygame.time.Clock()
            while True:
                clock.tick_busy_loop(60)
                if controller.parse_events(client, world, clock):
                    return
                world.tick(clock)
                world.render(display)
                # 시나리오 관련
                if world.scenario.is_running:
                    world.scenario.run_scenario()
                pygame.display.flip()

        finally:

            if (world and world.recording_enabled):
                client.stop_recorder()

            if world is not None:
                world.destroy()
                # spawned_prop.destroy()
                print("remove success")


            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
