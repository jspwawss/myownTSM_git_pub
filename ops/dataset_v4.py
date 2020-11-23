# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
import torchvision
import torch

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import csv
import ast

count = 0
class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        #print("row=",row)

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class YouCookDataSetRcg(data.Dataset):
    def __init__(self, root_path = r'/home/share/YouCook',list_file=r"/home/share/YouCook",
                num_segments=8, new_length=1, modality="RGB",
                train=False,test=False,val=False,
                vtmpl = r"{}.mp4", ttmpl=r"{}.txt", ftmpl=r"image_{}.jpg",
                slice=False, recognition=False,
                transforms=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((256,256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    

                    ]),
                    inputsize = 256,
                ):
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        #self.id2labelDict=None
        #self.URL2titleDict=None
        self.train=train
        self.test=test
        self.val=val
        self.vtmpl = vtmpl
        self.ttmpl = ttmpl
        self.ftmpl = ftmpl
        self.inputsize = inputsize
        self.transforms = transforms
        #self.slice = slice
        #self.recognition= recognition


        self.tmpFilename = None
        self.VideoTimeStamps = None
        if self.val ^ self.train == False:
            print("neither train nor tets")
            #return False
        if self.train:
            self.mode = "train"
        elif self.val:
            self.mode = "val"
        elif self.test:
            self.mode = "test"
        
        #print(self.mode)
        #exit()
        self.llist=[]
        #self.traininglist =[]
        #self.validationlist=[]
        self._label2Dict()
        self._getAnnotation()

    def _getAnnotation(self):#only include train(1333) and val(457) = 1790
        with open(os.path.join(self.root_path,"annotations","youcookii_annotations_trainval.txt"),"r") as _anntxt:
            text = _anntxt.read()
            d = ast.literal_eval(text)
            self.database = d["database"]
            for key, value in self.database.items():
                #print(key)
                #print(len(value['annotations']))
                #exit()
                if self.mode in value["subset"]:
                    for i in range(len(value["annotations"])):

                        self.llist.append([key,value["recipe_type"],value["annotations"][i]])
                #print( self.traininglist)
                #exit()
            #print(len(self.llist))
            #exit()
            #print(len(self.traininglist))           #10337  [[URL,annotations], [], ...]
            #print(len(self.validationlist))         #3492
            pass

                               
    def _takeSeg(self,_segment):
        return _segment[0][1]-_segment[0][0]
    def _takeid(self,_id):
        return _id[2]
    def _getSentence(self,id:str = None, URL:str=None,segment:list=None): #get annotations sentence
                                        # will call _getVideo to get video
                                        # return [[segment,sentence,id,videos], ...     ]

        rst=[]
        #items=self.database[URL]

        #annotations = items["annotations"]
        try:    #successful download from youtube

            start = segment[0]
            end = segment[1]
            #start = 2
            #end = 4
            if end - start +1 >=8:
                space = (end-start+1)//8
                allocate = [start + i*space for i in range(8)]
            else:
                space = 8 // (end-start+1)
                allocate=[]
                i = space
                while len(allocate)!=8:
                    if i < 0:
                        start +=1
                        i = space
                    allocate.append(start)
                    i -=1
            #print(allocate)

            videos = []
            print("finish allocation numbers")
            for a in allocate:
                video = self._getVideo(id=id, URL=URL, idx = a)
                videos.append(video)
                print(len(videos)) 
            #exit()
            return videos

        except Exception as e:
            print(e)
            return -1

    def _getVideo(self,id:str=None, URL:str=None, idx:str=None):    #get video clip in caption time
                            #return [video clip]

        print("id={},url={}".format(type(id),type(URL)))
        filename = os.path.join(self.root_path,"image_v2",str(id),URL,self.ftmpl.format(str(idx)))
 
        print(filename,"is ",os.path.isfile(filename))
        #return torch.tensor(np.zeros((1,3,256,256)))
        try:
            videoStream = Image.open(filename)
        except:
            return -1
        onlyVideo = videoStream
        
        #onlyVideo = onlyVideo.transpose(1,3)
        onlyVideo = np.swapaxes(onlyVideo, 0,2)
        oriH, oriW = np.shape(onlyVideo)[-2], np.shape(onlyVideo)[-1]
        #oriH, oriW = onlyVideo.size()[-2], onlyVideo.size()[-1]
        if min(oriH,oriW) >= self.inputsize:
            #print("ayayay")
            onlyVideo = onlyVideo[...,(oriH//2) -(min(oriH,oriW)//2) : (oriH//2)+(min(oriH,oriW)//2), oriW//2-min(oriH,oriW)//2:oriW//2+min(oriH,oriW)//2]
            #onlyVideo = onlyVideo[...,::min(oriH,oriW)/self.inputsize,::min(oriH,oriW)/self.inputsize]
            cropVideo = np.empty([1,3,self.inputsize,self.inputsize])
            for i in range(self.inputsize):
                for j in range(self.inputsize):
                    cropVideo[...,i,j] = onlyVideo[...,int(min(oriH,oriW)/self.inputsize*i),int(min(oriH,oriW)/self.inputsize*j)]
            #print(np.shape(cropVideo))

        else:
            #print("ayayay")
            onlyVideo = onlyVideo[...,(oriH//2) -(min(oriH,oriW)//2) : (oriH//2)+(min(oriH,oriW)//2), oriW//2-min(oriH,oriW)//2:oriW//2+min(oriH,oriW)//2]
            #onlyVideo = onlyVideo[...,::min(oriH,oriW)/self.inputsize,::min(oriH,oriW)/self.inputsize]
            cropVideo = np.empty([1,3,self.inputsize,self.inputsize])
            for i in range(self.inputsize):
                for j in range(self.inputsize):
                    cropVideo[...,i,j] = onlyVideo[...,int(min(oriH,oriW)/self.inputsize*i),int(min(oriH,oriW)/self.inputsize*j)]

        print("---after readvideo----")
        
        return torch.tensor(cropVideo)  #origin shape[frames,360,640,3] -->[frames, 3, 640, 360] -->[frames, 3, 224,224] and not normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        pass                

    def _getCaptions(self,id:str=None, URL:str=None): #get video captions
                               #return [[captions, start time, endtime],[captions, start, end], ...]
        #print("in _getCaptions")
        #print(os.path.join(self.root_path,"captions",species,self.ttmpl.format(title)),"is ",os.path.isfile(
        #    os.path.join(self.root_path,"captions",species,self.ttmpl.format(title))))
        capRes = []
        with open(os.path.join(self.root_path,"captions_v2",species,self.ttmpl.format(title)),"r") as captxt:
            lines = captxt.read().strip().split("\n\n")

            for line in lines:
                #print(line)
                #exit()
                line = line.split("\n")
                start = line[1].split("-->")[0]
                end = line[1].split("-->")[1]
                captions = line[2]
                capRes.append([captions,start,end])
        return capRes

    def _label2Dict(self): #convert id to name
                            #and get dictionary from name to label
        with open(os.path.join(self.root_path,"label_foodtype.csv"),"r") as _csv:
        
            lines = csv.reader(_csv,delimiter=",")
            
            self.getLabelDict = dict((line[0],idx) for idx, line in enumerate(lines))

        pass

    def __len__(self):  #for dataset iter need
        return len(self.llist)
    def __getitem__(self, index):   #for dataset iter need
                                    #return [label, id, URL, annotation, video clips]
        #self.index = index
        
        item = self.llist[index]
        #print(item)
        URL = item[0]   #aFsdfsd464
        id = item[1]    #100~130,200~230,300~325,400...
        
        annotation = item[2]["sentence"]
        segment = item[2]["segment"]
        #print(type(segment[0]))
        #exit()
        try: #download successful

            label = self.getLabelDict[id] #predict label 0~100
            videos = self._getSentence(id=id, URL=URL, segment=segment)
            #print("finish loading")
            if len(videos) == 1:
                print("videos DNE")
                return -1
                exit()
            res = []
            res.append([label,id,URL,annotation,videos])
            #res.append([URL])
            #print(res)

            #exit()
            return res                      
        except Exception as e:
            print(e)

            if index >2:

                exit()
            return False

        
        pass


if __name__ == "__main__":
    crop_size = 224
    dataset = YouCookDataSetRcg(train=True,inputsize=crop_size)
    for i in dataset:
        print(i)
        #exit()