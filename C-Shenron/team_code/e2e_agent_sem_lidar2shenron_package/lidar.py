import sys
import os
# sys.path.append("../")
sys.path.append('radar-imaging-dataset/carla_garage_radar/team_code/e2e_agent_sem_lidar2shenron_package/')

import numpy as np
from e2e_agent_sem_lidar2shenron_package.path_config import *
from e2e_agent_sem_lidar2shenron_package.ConfigureRadar import radar
from e2e_agent_sem_lidar2shenron_package.shenron.Sceneset import *
from e2e_agent_sem_lidar2shenron_package.shenron.heatmap_gen_fast import *
# from pointcloud_raytracing.pointraytrace import ray_trace
import scipy.io as sio
from e2e_agent_sem_lidar2shenron_package.lidar_utils import *
import time
import shutil
import pdb

def map_carla_semantic_lidar_latest(carla_sem_lidar_data):
    '''
    Map CARLA 0.9.16 semantic LiDAR tags to Shenron radar simulator input classes.

    CARLA 0.9.16 CityObjectLabel IDs:
      0=NONE, 1=Roads, 2=Sidewalks, 3=Buildings, 4=Walls, 5=Fences,
      6=Poles, 7=TrafficLight, 8=TrafficSigns, 9=Vegetation, 10=Terrain,
      11=Sky, 12=Pedestrians, 13=Rider, 14=Car, 15=Truck, 16=Bus,
      17=Train, 18=Motorcycle, 19=Bicycle, 20=Static, 21=Dynamic,
      22=Other, 23=Water, 24=RoadLines, 25=Ground, 26=Bridge,
      27=RailTrack, 28=GuardRail

    Shenron material classes (consumed by new_map_material in lidar_utils.py):
      0=road/ground (suppressed), 1=sidewalk->concrete, 2=building->concrete,
      3=wall->concrete, 4=fence->metal, 5=pole->metal, 6=traffic_light->metal,
      7=traffic_sign->metal, 8=vegetation->wood, 9=terrain->wood,
      10=sky->0, 11=person->human, 12=rider->metal, 13=car->metal,
      14=truck->metal, 15=bus->metal, 16=train->metal, 17=motorbike->metal,
      18=bicycle->metal
    '''
    carla_sem_lidar_data_crop = carla_sem_lidar_data[:, (0, 1, 2, 5)]

    # Index = CARLA 0.9.16 tag ID, Value = Shenron internal class
    temp_list = np.zeros(256, dtype=np.int32)
    temp_list[0]  = 0   # NONE       -> road/ground (suppressed)
    temp_list[1]  = 0   # Roads      -> road/ground (suppressed)
    temp_list[2]  = 1   # Sidewalks  -> sidewalk/concrete
    temp_list[3]  = 2   # Buildings  -> building/concrete
    temp_list[4]  = 3   # Walls      -> wall/concrete
    temp_list[5]  = 4   # Fences     -> fence/metal
    temp_list[6]  = 5   # Poles      -> pole/metal
    temp_list[7]  = 6   # TrafficLight -> traffic_light/metal
    temp_list[8]  = 7   # TrafficSigns -> traffic_sign/metal
    temp_list[9]  = 8   # Vegetation -> vegetation/wood
    temp_list[10] = 9   # Terrain    -> terrain/wood
    temp_list[11] = 10  # Sky        -> sky (suppressed)
    temp_list[12] = 11  # Pedestrians -> person/human
    temp_list[13] = 12  # Rider      -> rider/metal
    temp_list[14] = 13  # Car        -> car/metal
    temp_list[15] = 14  # Truck      -> truck/metal
    temp_list[16] = 15  # Bus        -> bus/metal
    temp_list[17] = 16  # Train      -> train/metal
    temp_list[18] = 17  # Motorcycle -> motorbike/metal
    temp_list[19] = 18  # Bicycle    -> bicycle/metal
    temp_list[20] = 0   # Static     -> suppressed
    temp_list[21] = 0   # Dynamic    -> suppressed
    temp_list[22] = 0   # Other      -> suppressed
    temp_list[23] = 0   # Water      -> suppressed
    temp_list[24] = 0   # RoadLines  -> road/ground (suppressed)
    temp_list[25] = 0   # Ground     -> road/ground (suppressed)
    temp_list[26] = 2   # Bridge     -> building/concrete
    temp_list[27] = 0   # RailTrack  -> suppressed
    temp_list[28] = 4   # GuardRail  -> fence/metal

    tags = carla_sem_lidar_data_crop[:, 3].astype(int)
    tags = np.clip(tags, 0, len(temp_list) - 1)
    col = temp_list[tags]
    carla_sem_lidar_data_crop[:, 3] = col

    return carla_sem_lidar_data_crop

# def map_carla_semantic_lidar(carla_sem_lidar_data):
#     '''
#     Function to map material column in the collected carla ray_cast_shenron to shenron input 
#     '''
#     # print(carla_sem_lidar_data.shape())
#     carla_sem_lidar_data_crop = carla_sem_lidar_data[:, (0, 1, 2, 5)]
#     carla_sem_lidar_data_crop[:, 3] = carla_sem_lidar_data_crop[:, 3]-1
#     carla_sem_lidar_data_crop[carla_sem_lidar_data_crop[:, 3]>18, 3] = 255.
#     carla_sem_lidar_data_crop[carla_sem_lidar_data_crop[:, 3]<0, 3] = 255.
#     carla_sem_lidar_data_crop[:, (0, 1, 2)] = carla_sem_lidar_data_crop[:, (0, 2, 1)]
#     return carla_sem_lidar_data_crop

def check_save_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return

def rotate_points(points, angle):
    rotMatrix = np.array([[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]
        , [- np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0]
        , [0, 0, 1]])
    return np.matmul(points, rotMatrix)

def Cropped_forRadar(pc, veh_coord, veh_angle, radarobj):
    """
    Removes Occlusions and calculates loss for each point
    """

    skew_pc = rotate_points(pc[:, 0:3] , veh_angle )
    # skew_pc = np.vstack(((skew_pc ).T, pc[:, 3], pc[:, 5])).T  #x,y,z,speed,material
    skew_pc = np.vstack(((skew_pc ).T, pc[:, 3], pc[:, 5],pc[:,6])).T  #x,y,z,speed,material, cosines

    rowy = np.where((skew_pc[:, 1] > 0.8))
    new_pc = skew_pc[rowy, :].squeeze(0)

    new_pc = new_pc[new_pc[:,4]!=0]

    new_pc = new_pc[(new_pc[:,0]<50)*(new_pc[:,0]>-50)]
    new_pc = new_pc[(new_pc[:,1]<100)]
    new_pc = new_pc[(new_pc[:,2]<2)]

    simobj = Sceneset(new_pc)

    [rho, theta, loss, speed, angles] = simobj.specularpoints(radarobj)
    # print(f"Number of points = {rho.shape[0]}")
    return rho, theta, loss, speed, angles

def run_lidar(sim_config, sem_lidar_frame):

    #restructed lidar.py code

    # lidar_path = f'{base_folder}/{sim_config["CARLA_SHENRON_SEM_LIDAR"]}'
    # # lidar_velocity_path = f'{base_folder}/{sim_config["LIDAR_PATH_POINT_VELOCITY"]}/'
    # out_path = f'{base_folder}/{sim_config["RADAR_PATH_SIMULATED"]}'

    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    # shutil.copyfile('ConfigureRadar.py',f'{base_folder}/radar_params.py')

    # lidar_files = os.listdir(lidar_path)
    # lidar_velocity_files = os.listdir(lidar_velocity_path)
    # lidar_files.sort()
    # lidar_velocity_files.sort()

    # print(lidar_files)
    
    #Lidar specific settings
    radarobj = radar(sim_config["RADAR_TYPE"])
    # radarobj.chirps = 128
    radarobj.center = np.array([0.0, 0.0])  # center of radar
    radarobj.elv = np.array([0.0])

    # setting the sem lidar inversion angle here
    veh_angle = sim_config['INVERT_ANGLE']
    
    # all_speeds = []

    # temp_angles = []
    # temp_rho = []
    # for file_no, file in enumerate(lidar_files):
        
    #     start = time.time()
    #     if file.endswith('.npy'):  # .pcd
    #         print(file)
            
    #         lidar_file_path = os.path.join(f"{lidar_path}/", file)
    #         load_pc = np.load(lidar_file_path)

            # load_velocity = np.load(f'{lidar_velocity_path}/{file}')

            # test = map_material(test)
    cosines = sem_lidar_frame[:, 3]
    load_pc = sem_lidar_frame
    load_pc = map_carla_semantic_lidar_latest(load_pc)
    test = new_map_material(load_pc)
    
    points = np.zeros((np.shape(test)[0], 7))
    # points[:, [0, 1, 2]] = test[:, [0, 2, 1]]
    points[:, [0, 1, 2]] = test[:, [1, 0, 2]]

    """
    points mapping
    +ve ind 0 == right
    +ve ind 1 == +ve depth
    +ve ind 2 == +ve height
    """
    # add the velocity channel here to the lidar points on the channel number 3 most probably
    # points[:, 3] = load_velocity

    points[:, 5] = test[:, 3]
    points[:, 6] = cosines
    ### if jason carla lidar
    # points = np.zeros((np.shape(test)[0], 6))
    # points[:, [0, 1, 2]] = load_pc[:, [0, 1, 2]]
    # points[:, 5] = 4
    ##########

    # if USE_DIGITAL_TWIN:
    #     gt_label = gt[file_no,:]
    #     points, veh_speeds = create_digital_twin(points, gt_label) ## This also claculates and outputs speed

    #     all_speeds.append(veh_speeds)

    # if sim_config["RADAR_MOVING"]:
    #     # when the radar is moving, we add a negative doppler from all the points
    #     if INDOOR:
    #         curr_radar_speed = radar_speeds[file_no,:]

    #         cos_theta = (points[:,1]/np.linalg.norm(points[:,:2],axis=1))
    #         radial_speed = -np.linalg.norm(curr_radar_speed)*cos_theta

    #         points[:,3] += radial_speed
    #         points[:,5] = 4 ## harcoded 

    
    Crop_rho, Crop_theta, Crop_loss, Crop_speed, Crop_angles = Cropped_forRadar(points, np.array([0, 0, 0]), veh_angle, radarobj)

    """ DEBUG CODE
    spec_angle_thresh = 2*np.pi/180#*(1/rho)

    print(f"Number of points < 2deg = {np.sum(abs(Crop_angles)<spec_angle_thresh)}")
    temp_angles.append(np.sum(abs(Crop_angles)<spec_angle_thresh))
    temp_rho.append(Crop_rho.shape[0])
    continue
    """
    

    if sim_config["RAY_TRACING"]:
        rt_rho, rt_theta = ray_trace(points)

        rt_loss = np.mean(Crop_loss)*np.ones_like(rt_rho)
        rt_speed = np.zeros_like(rt_rho)
        Crop_rho = np.append(Crop_rho, rt_rho)
        Crop_theta = np.append(Crop_theta, rt_theta)
        Crop_loss = np.append(Crop_loss, rt_loss)
        Crop_speed = np.append(Crop_speed, rt_speed)

    adc_data = heatmap_gen(Crop_rho, Crop_theta, Crop_loss, Crop_speed, radarobj, 1, 0)
    return adc_data
    # check_save_path(out_path)
    # np.save(f'{out_path}/{file[:-4]}', adc_data)
    # diction = {"adc_data": adc_data}
    # sio.savemat(f"{out_path}/{file[:-4]}.mat", diction)
    # sio.savemat(f"test_pc.mat", diction)
    # print(f'Time: {time.time()-start}')
    # np.save("all_speeds_no_micro.npy",np.array(all_speeds))
    """ DEBUG CODE
    fig, ax = plt.subplots(1,2)
    ax[0].plot(temp_angles)
    ax[1].plot(temp_rho)
    
    plt.plot(temp_rho)
    plt.show()
    pdb.set_trace()
    """ 

if __name__ == '__main__':

    points = np.zeros((100,6))

    points[:,5] = 4
    
    points[:,0] = 1
    points[:,1] = np.linspace(0,15,100)
    
    points[:,3] = -0.5*np.cos(np.arctan2(points[:,0],points[:,1]))
    radarobj = radar('radarbook')
    # radarobj.chirps = 128
    radarobj.center = np.array([0.0, 0.0])  # center of radar
    radarobj.elv = np.array([0.0])

    Crop_rho, Crop_theta, Crop_loss, Crop_speed = Cropped_forRadar(points, np.array([0, 0, 0]), 0, radarobj)
    Crop_loss = np.ones_like(Crop_loss)
    adc_data = heatmap_gen(Crop_rho, Crop_theta, Crop_loss, Crop_speed, radarobj, 1, 0)
    diction = {"adc_data": adc_data}
    sio.savemat(f"test_pc.mat", diction)
