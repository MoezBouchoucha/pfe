import warnings
warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs")
from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
    
suppress_qt_warnings()


from download import doawnload_manager 
from ml import model
import os 
import random
import numpy as np
import open3d as o3d
import trimesh
import pyrender
from skimage.transform import resize
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg















def ply_to_obj(ply_path, dest_path,methode="ball"):
    
    pcd = o3d.io.read_point_cloud(ply_path)
    
    pcd.estimate_normals()

    if methode=="ball":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 2 * avg_dist   

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2]))
    else:
        mesh,densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=8,
                linear_fit=True,
                n_threads=1)

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                              vertex_normals=np.asarray(mesh.vertex_normals))

    tri_mesh.export(dest_path, "obj")

    return dest_path

def obj_to_ply(path,num_points):
       mesh=o3d.io.read_triangle_mesh(path,True)
       #print("*"*10,path[:len(path)-3]+"mtl")
       #    o3d.io.read_mtl(path[:len(path)-3]+"mtl", mesh) 
       #o3d.visualization.draw([mesh]) # type: ignore
       pcd = mesh.sample_points_uniformly(num_points)
       return numpy_to_ply(np.asarray(pcd.points),path[:len(path)-3]+"ply")

def numpy_to_ply(points, ply_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(ply_path, pcd)
    return ply_path

def rescali(pcd,size):
    num_points_x = size[0]
    num_points_y = size[1]
    num_points_z = size[2]
    
    x_voxel_size = (pcd.get_max_bound()[0] - pcd.get_min_bound()[0]) / num_points_x
    y_voxel_size = (pcd.get_max_bound()[1] - pcd.get_min_bound()[1]) / num_points_y
    z_voxel_size = (pcd.get_max_bound()[2] - pcd.get_min_bound()[2]) / num_points_z
    
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=[x_voxel_size, y_voxel_size, z_voxel_size])
    
    return downsampled_pcd






#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


class loader:
    def __init__(self,img_target_size,obj_target_size):
        self.loaded=[]
        self.img_target_size=img_target_size
        self.obj_target_size=obj_target_size
        self.list_objects=[]
        self.dm=doawnload_manager()
        self.model=model()
    #######################################################################################################
    def helper(self):
        print("*"*100,"\nel class hedhy tloady ay 7aja ettelechargeha mel class download w t7ot el pipeline mte3 el training(architecture mte3 el gan fi el train ....) ")
        
        print("Normalment esm el 2d howa bidou esm el 3d \t exmpl : 001.png ==> 001.pcd\n","*"*100)
    #######################################################################################################
    def load_img(self,path):
        return(resize(mpimg.imread(path), self.img_target_size))
    #######################################################################################################
    def load_pcd(self,path):
        pcd=o3d.io.read_point_cloud(path)
        # pcd.sample_points_uniformly(self.obj_target_size)
        return np.asarray(pcd.points)
    #######################################################################################################
    def load_random_path(self):
        while len(self.list_objects)>=len(self.loaded):
            path_choix=random.randint(0,len(self.list_objects))
            if path_choix not in self.loaded :
                self.loaded.append(path_choix)
                return(path_choix)
        return(-1)
    
    #######################################################################################################
    def downloads(self,batch_size):
        return self.dm.load_batch(batch_size=batch_size)
    
    #########################################################################################################

    def remove_li_downloaditou(self,paths):
        for i in paths:
            if os.path.exists(i):
                os.remove(i)
            else:
                print(f"The file {i} does not exist")

    #########################################################################################################
    
    def split_train_valid(self,x,y,ratio=0.2):
        nbr_train=len(x)-(len(x)*ratio)
        
        x_train=x[:nbr_train]
        y_train=y[:nbr_train]
        x_valid=x[nbr_train:]
        y_valid=y[nbr_train:]
        
        return x_train,y_train,x_valid,y_valid
    
    #########################################################################################################
    def preprocess_data(self,img,pcd):
        l=[]
        for i in img :
            #el image lezemha twalli array se33a 
            # image=np.asarray(img)
            #resize image
            image=resize(i,(self.img_target_size[0],self.img_target_size[1],3))
            # print(image.shape,type(image),min(image.flatten()),max(image.flatten()))
            # plt.imshow(image)
            # plt.show()

            # normalize image
            # image=image/255.0 # type: ignore
            # print("image shape =",image.shape," wanted shape =",self.img_target_size)
            l.append(image)
        # print("#"*10,"\n","#"*10)
        l=np.asarray(l)
        pcd=np.asarray(pcd)
        return l,pcd
    
    #########################################################################################################
    
    def custom_loader(self,valeur):

        if valeur[0]==0:
            return self.redwood_loader(valeur[1])
        if valeur[0]==1:
            return self.shapenet_loader(valeur[1])
        else :
            print("passs")

    #########################################################################################################

    def pcd2img(self,path,vizualizi=False):
        point_cloud = o3d.io.read_point_cloud(path)
        if vizualizi:
            o3d.visualization.draw([point_cloud]) # type: ignore
        
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_zyx((-2*np.pi-np.pi/2, 2*np.pi/3+np.pi/6, np.pi/2))
        point_cloud.rotate(rotation_matrix)
        vis = o3d.visualization.Visualizer() # type: ignore  
        
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer())
        vis.destroy_window()
        return image


    #########################################################################################################
    def redwood_loader(self,path):
        pcd=o3d.io.read_point_cloud(path)
        scale=np.asarray(pcd.points).shape[0]//self.obj_target_size

        # pcd =pcd.uniform_down_sample(int(scale))

        pcd = pcd.uniform_down_sample(int(scale))


        indices = np.random.choice(len(pcd.points), size=self.obj_target_size, replace=False)
        pcd  = pcd.select_by_index(indices)

        pcd=np.asarray(pcd.points)
        print(pcd.shape)
        
        img=self.pcd2img(path,True)
        
        return img,pcd
    #########################################################################################################
    def obj2img(self,path):
        obj = trimesh.load(path,force="mesh")

        #theta = np.random.uniform(0, 2*np.pi)
        #quat = trimesh.transformations.quaternion_about_axis(theta, [0, 0, 1])
        #obj.apply_transform(trimesh.transformations.quaternion_matrix(quat)) # type: ignore

        mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)
        
        
        # compose scene
        scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[random.randint(0,256),random.randint(0,256),random.randint(0,256)])
        camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

        scene.add(mesh, pose=  np.eye(4))
        scene.add(light, pose=  np.eye(4))
        
        c = 2**-0.5
        scene.add(camera, pose=[[ 1,  0,  0,  0],   
                            [ 0,  c, -c, -2],
                            [ 0,  c,  c,  2],
                            [ 0,  0,  0,  1]])

        # render scene
        r = pyrender.OffscreenRenderer(1080,1080)
        color, _ = r.render(scene) # type: ignore

        return color

        
    #########################################################################################################
    def shapenet_loader(self,path):
        pcd=o3d.io.read_point_cloud(obj_to_ply(path,num_points=self.obj_target_size))
        pcd=np.asarray(pcd.points)
        img=self.obj2img(path)
        return img,pcd
    #######################################################################################################
    def load_data(self,valeurs):
        img_batch=[]
        obj_batch=[]
        for i in valeurs:
            img,pcd = self.custom_loader(i)  # type: ignore
            img_batch.append(img)
            obj_batch.append(pcd)
        return img_batch,obj_batch
    #########################################################################################################
    def train(self,model,num_epochs,batch_size):
        
        for i in range(num_epochs):
            print("Started epoch {}/{} :{}".format(i,num_epochs,str((i+1)*100/num_epochs)[:6]))
            
            # paths=self.downloads(batch_size)
            paths=[[1,".\\data\\shapenet\\02876657\\02876657\\1a7ba1f4c892e2da30711cdbdbc73924\\model.obj"],[1,r"data\shapenet\03046257\03046257\1a157c6c3b71bbd6b4792411407bb04c\model.obj"]]

            img,pcd=self.load_data(paths)

            print(f"pcd shape {len(pcd)} , pcd type {type(pcd)}")
            print(f"img shape {len(img)} , img type {type(img)}")
            img,pcd=self.preprocess_data(img,pcd)


            print(f"pcd shape {pcd.shape} , pcd type {type(pcd)}")
            print(f"img shape {img.shape} , img type {type(img)}")

            print("so far soo good ...")
            # break
            shape=self.obj_target_size

            d_model=model.define_descriminator(shape)
            g_model=model.define_generator(shape)
            gan_model=model.define_gan(g_model,d_model,shape)

            
            x_train,y_train,x_valid,y_valid=self.split_train_valid(img,pcd,0.2)
            
            losses_train=model.train_for_one_epoch(x_train,y_train,g_model,d_model,gan_model,num_epochs,batch_size)

            #print(f"\n Epoch {i}: Train loss {str(losses_train_mean)[:5]} Validation loss {str(losses_valid_mean)[:5]}")
            self.remove_li_downloaditou(paths)
            break
        pass




# a=loader((500,600,3),1000)
# #a.helper()
# a.dm.dynamic_current_data()
# a.dm.loaded=a.dm.current_data
# a.dm.fin_dataset()
# a.train(1,1,1)
# img,pcd=a.load_data(a.downloads(batch_size=1))

# x,y=a.shapenet_loader("data\\shapenet\\03759954\\03759954\\1a2e3602e54e8e5f4fdb7f619836e43d\\model.obj")
# img,pcd=a.redwood_loader("data\\mesh\\10664.ply")
# print(pcd.shape)
# plt.imshow(img)
# plt.show()
# img,pcd=a.load_data([[1,'data\\shapenet\\03337140\\03337140\\1a0995936b34c17367aa983983f9bf36\\model.obj']])
#a.remove_li_downloaditou(test)
# print(a.dm.loaded,a.dm.batch)


