import PIL.Image as img
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from scipy.spatial.transform import Rotation as R

def gcopVecToUnity(v):
    temp = np.array((v[0],v[2],v[1]))
    return temp

def makeCamAct(x):
    #Set up camera action
    if (len(x) is 2) or (len(x) is 4):
        cameraPos = x[0]
        r = R.from_dcm(x[1])
    elif (len(x) is 6):
        cameraPos = x[0:3]
        r = R.from_euler('zyx',x[3:6])
    else:
        print("Unsupported x format")
        return
    euler = r.as_euler(seq = 'zxy',degrees = True) #using weird unity sequence
    unityEuler = (euler[1],-euler[0],euler[2]) 
    cameraAction = np.hstack((gcopVecToUnity(cameraPos),unityEuler,1))
    return np.array([cameraAction])

def save_image_array(image_in,path,name):
  im_np = (image_in*255).astype('uint8')
  image = img.fromarray(im_np)
  if path is not None:
      image.save(path + name + '.png')
  else:
      image.show()

def unity_image(env,act,cam_name,env_name=None):
    obs = None
    if cam_name is not None:
        env.set_actions(cam_name,act)
        env.step()
        (ds,ts) = env.get_steps(cam_name)
        obs = ds.obs[0][0,:,:,:]
    envobs = None
    if env_name is not None:
        env.set_actions(env_name,np.zeros((1,4)))
        env.step()
        (ds,ts) = env.get_steps(env_name)
        envobs = ds.obs[0][0,:,:,:]
    if envobs is None:
        return obs
    if obs is None:
        return envobs
    return (obs,envobs)

def move_orange(env,orangeName,orangePos,orangeColor,treePos,collisionR,camName=None,spawn_orange=True):
    dx = orangePos[0:3] - treePos[0:3]
    theta = np.arctan2(dx[1],dx[0])
    r_min = 0
    r_max = 3
    eps = 0.01
    orange_offset = 0.1 + collisionR
    orange_h_offset = 0.2
    while (r_max-r_min) > eps:
        r = (r_max + r_min)/2
        tempPos = treePos + np.array([np.cos(theta)*r,np.sin(theta)*r,dx[2]+orange_h_offset])
        orangeAct = np.array([np.hstack((gcopVecToUnity(tempPos),orangeColor,1))])
        env.set_actions(orangeName,orangeAct)
        env.step()
        (ds,ts) = env.get_steps(orangeName)
        if np.linalg.norm(ds.obs[0][0,0:3]-ds.obs[0][0,3:6]) < eps:
            r_min = r
        else:
            r_max = r
    if (camName is None):
        r = (r_max + r_min)/2 - collisionR + orange_offset
        tempPos = treePos + np.array([np.cos(theta)*r,np.sin(theta)*r,dx[2]+orange_h_offset])
        if not spawn_orange:
            tempPos = np.array([0.0, 0.0, 0.0])
        orangeAct = np.array([np.hstack((gcopVecToUnity(tempPos),orangeColor,1))])
        env.set_actions(orangeName,orangeAct)
        env.step()
        return orangeAct, None
    else:
        threshold = 0.1
        reference_r = 1
        w = (140,240,270,370)
        tempPos = treePos + np.array([np.cos(theta)*r,np.sin(theta)*r,dx[2]+orange_h_offset])
        orangeAct = np.array([np.hstack((gcopVecToUnity(tempPos),orangeColor,1))])
        env.set_actions(orangeName,orangeAct)
        env.step()
        tempX = treePos + np.array([np.cos(theta)*(r+reference_r),np.sin(theta)*(r+reference_r),dx[2]+orange_h_offset])
        tempX = np.hstack((tempX,np.pi+theta,0,0))
        camAct = makeCamAct(tempX)
        image_arr = unity_image(env,camAct,camName)
        #save_image_array(image_arr,None,None)
        image_trunc = image_arr[w[0]:w[1],w[2]:w[3],0]
        #save_image_array(image_trunc,None,None)
        image_trunc = (image_trunc > threshold)*image_trunc
        #save_image_array(image_trunc,None,None)
        unobscured_score = np.count_nonzero(image_trunc > threshold)
        r = r - collisionR
        tempPos = treePos + np.array([np.cos(theta)*(r),np.sin(theta)*(r),dx[2]+orange_h_offset])
        if not spawn_orange:
            tempPos = np.array([0.0, 0.0, 0.0])
        orangeAct = np.array([np.hstack((gcopVecToUnity(tempPos),orangeColor,1))])
        env.set_actions(orangeName,orangeAct)
        env.step()
        orangeAct = treePos + np.array([np.cos(theta)*(r+orange_offset),np.sin(theta)*(r+orange_offset),dx[2]+orange_h_offset])
        #orangeAct = np.array([np.hstack((gcopVecToUnity(tempPos),orangeColor,1))])

        tempX = treePos + np.array([np.cos(theta)*(r+reference_r),np.sin(theta)*(r+reference_r),dx[2]+orange_h_offset])
        tempX = np.hstack((tempX,np.pi+theta,0,0))
        camAct = makeCamAct(tempX)
        image_arr = unity_image(env,camAct,camName)
        #save_image_array(image_arr,None,None)
        image_trunc = image_arr[w[0]:w[1],w[2]:w[3],0]
        #save_image_array(image_trunc,None,None)
        image_trunc = (image_trunc > threshold)*image_trunc
        #save_image_array(image_trunc,None,None)
        obscured_score = np.count_nonzero(image_trunc > threshold)
        occlusion_ratio = 1 - (obscured_score/unobscured_score)
        print("Occluded: ", occlusion_ratio*100, "%")
        return orangeAct, occlusion_ratio
