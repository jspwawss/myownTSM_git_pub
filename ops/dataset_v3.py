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
    def __init__(self, root_path,list_file,
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
            return False
        if self.train:
            self.mode = "train"
        elif self.val:
            self.mode = "val"
        elif self.test:
            self.mode = "test"
        self._getList(mode=self.mode)
        #print(self.mode)
        #exit()
 
        self._label2Dict()
        self._getAnnotation()

    def _getAnnotation(self):#only include train(1333) and val(457) = 1790
        with open(os.path.join(self.root_path,"annotations","youcookii_annotations_trainval.txt"),"r") as _anntxt:
            text = _anntxt.read()
            d = ast.literal_eval(text)
            self.database = d["database"]
            
            pass

    def _getList(self,mode:str=None):
        with open(os.path.join(self.root_path,"split",mode+"_list.txt")) as listtxt:
            self.llist = listtxt.readlines()
        

                               
    def _takeSeg(self,_segment):
        return _segment[0][1]-_segment[0][0]
    def _takeid(self,_id):
        return _id[2]
    def _getSentence(self,id:str = None, URL:str=None): #get annotations sentence
                                        # will call _getVideo to get video
                                        # return [[segment,sentence,id,videos], ...     ]

        rst=[]
        items=self.database[URL]

        annotations = items["annotations"]
        try:    #successful download from youtube
            totalFrames = 0
            for annotation in annotations:
                segment=annotation["segment"]
                totalFrames += segment[1]-segment[0]+1
                id = annotation["id"]
                sentence = annotation["sentence"]
                rst.append([segment,sentence,id])

            while len(rst) >8:  #we only input 8 frames
                _min=1000
                _minidx = None
                for idx, r in enumerate(rst):
                    if (r[0][1]-r[0][0]) < _min:
                        _min = r[0][1]-r[0][1]
                        _minidx = idx
                totalFrames -= _min
                rst.pop(_minidx)

            rst.sort(key=self._takeSeg, reverse=True)

            allFrames = 8
            idx = 0

            allocate = [0] * len(rst)

            while allFrames >0:

                allocate[idx] +=1
                idx = (idx +1) % len(rst)
                allFrames -=1
            print(allocate)
            print("finish allocation numbers")
            for idx, r in enumerate(rst):
                spacing = (r[0][1]-r[0][0]) //(allocate[idx]+1)
  
                frames = [r[0][0] + spacing*f for f in range(1,allocate[idx]+1) ]
                videos = []

                for frame in frames:
                    video = self._getVideo(id=id, URL=URL, idx = frame)
                    #print(video)
                    #print(video.size())
                    #exit()
                    print("success get vidoe clip")
                    videos.append(video)
                    #print(videos)
                    ######do not comment it
                    print(len(videos)) ###use for check video return, if video return -1 will jump to exception
                    #### do not delete it

                rst[idx].append(videos)


            

            rst.sort(key=self._takeid, )
            print("rst\n",rst)
 
            return rst
        except Exception as e:
            print(e)
            return -1

    def _getVideo(self,id:str=None, URL:str=None, idx:int=None):    #get video clip in caption time
                            #return [video clip]

        print("id={},url={}".format(type(id),type(URL)))
        filename = os.path.join(self.root_path,"image_v2",str(id),URL,self.ftmpl.format(str(idx)))
 
        print(filename,"is ",os.path.isfile(filename))
        return torch.tensor(np.zeros((1,3,256,256)))
        videoStream = Image.open(filename)
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
        with open(os.path.join(self.root_path,"captions",species,self.ttmpl.format(title)),"r") as captxt:
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
                                    #return [label, URL, [[segment,sentence,id,videos], [], [], ...]
        #self.index = index
        item = self.llist[index].split("\n")[0]

        id = item.split("/")[0]    #100~130,200~230,300~325,400...
        URL = item.split("/")[1]   #aFsdfsd464

        try: #download successful

            label = self.getLabelDict[id] #predict label 0~100

            res = []
            res.append([label])
            res.append([URL])
            sentences = self._getSentence(id=id, URL=URL)
            if len(sentences) == 1:
                return -1
            res.append(sentences)

            return res                      
        except Exception as e:
            print(e)

            #if index >2:

            #    exit()
            return False

        
        pass


