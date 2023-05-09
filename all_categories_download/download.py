import redwood_3dscan as red 
import random 
import json
import shapenet

###############################################################
##########  badel el path  mte3 el json files #################
###############################################################

def _load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

#######################################################################################################
class doawnload_manager:
    def __init__(self):
        self.loaded=[] #el paths li lodithom fi el dataset li ne5dem 3leha taw
        self.list_datasets=["redwood","shapenet"] #lista mte3 el datasets
        self.current_download=0 #hedhy just reference l ama dataset ndownloadi taw 
        self.current_data=[] #hedhy just n7ot feha les path mte3 el dataset li ne5dem 3leha taw <==> ya3ni el liste des elements
        self.batch=[] #hedhy feha el batch elli nraj3ou kol marra

    #######################################################################################################

    def helper(self):
        print("*"*100,"\n el classe hedhy ettelechargi kol object 7achtek bih b facon dynamique , ya3ni dima tloadi 7aja ma 5demch beha 9bal w jdida w ken el data li di el redwood mithel wfet yzid b9eyet el batch mel co3d wenty mechy...")
        
        print("El data bech tkoun fi el structure hedhy: \t ./data==>/rgb /mesh\n esm el image howa bidou esm el mesh bark el extension moch kifkif \n","*"*100)
        
    #######################################################################################################
    def fin_dataset(self):
        self.loaded=[]
        self.current_download =self.current_download + 1
        print("#"*10,f"the current data loading is {self.list_datasets[self.current_download]} dataset","#"*10)

    #######################################################################################################
    def download(self,path):
        if self.current_download==0 :
            self.redwood_download(path)
        elif self.current_download==1:
            self.shapenet_download(path)
        else:
            print("passsss")

    #######################################################################################################
    def dynamic_current_data(self):
        if self.current_download==0 :
        
            self.current_data=red.meshes
        
        elif self.current_download ==1:

            self.current_data=shapenet.ids
        
        elif self.co3d_download==2:
            # a=_load_json(".\co3d\links.json")
            
            # l=[]
            # for i in a.keys():
            #     for j in a[i].keys():
            #         l.extend(a[i][j])
            # self.current_data=l
            pass
        else:
            pass
        
            
    #######################################################################################################
    def random_non_loaded_object(self):
        while 1:
            path_choix=random.choice(self.current_data)
            if path_choix not in self.loaded :
                self.loaded.append(path_choix)
                return path_choix

    #######################################################################################################
    def redwood_download(self,path):
        red.download_mesh(path)
        #red.download_video(path)
    #######################################################################################################
    def co3d_download(self):
        pass
    #######################################################################################################
    def shapenet_download(self,id):
        shapenet.downloader(id)
    
    #######################################################################################################
    def full_path(self,path):
        if self.current_download==0:
            return ".\\data\\mesh\\"+path+".ply"
        if self.current_download==1:
            return ".\\data\\shapenet\\"+path+"\\"+"model.obj"
    #######################################################################################################
    def load_batch(self,batch_size):
        self.batch=[]
        #loadi el path w telechargi el data
        for i in range(batch_size):
            
            self.dynamic_current_data()

            path=self.random_non_loaded_object()
            #print(path)
            print("*"*50)
            self.download(path)
            print("*"*50)
            self.batch.append([self.current_download,self.full_path(path)])

            #nchouf ken el data li ne5dem beha taw nejem nloadi menha 7aka o5ra walla le 
            if (len(self.loaded) >= len(self.current_data)):
                self.fin_dataset() 

        return self.batch



    

# a=doawnload_manager()
# a.dynamic_current_data()
# a.loaded=a.current_data[:-1]
# print(a.load_batch(2))
#print(a.batch)
# print(a.load_batch(1) ,len(a.batch))