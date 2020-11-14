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
                vtmpl = r"{}.mp4", ttmpl=r"{}.txt", ftmpl=r"image_{},jpg",
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
        self._URL2title()
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
        
    def _URL2title(self):   #convert URL 2 title name
        with open(os.path.join(self.root_path,"url2label.txt"),"r") as _txt:
            lines = _txt.readlines()
            #print("lines=",lines)
            self.URL2titleDict = dict((line.split("\t")[0].strip(),line.split("\t")[1].strip()) for line in lines)
        pass
                                            
    def _takeSeg(self,_segment):
        return _segment[0][1]-_segment[0][0]
    def _takeid(self,_id):
        return _id[2]
    def _getSentence(self,URL:str=None): #get annotations sentence
                                        # will call _getVideo to get video
                                        # return [[segment,sentence,id,videos], ...     ]

        #print("in _getSentence")
        rst=[]
        #print(self.database)
        #print("URL=",URL)
        #print(self.database)
        items = self.database[URL]
        #print("items=",items)
        title = self.URL2titleDict[URL]
        #print("title=",title)
        recipe_type = items["recipe_type"]
        species = self.id2labelDict[recipe_type]
        label = self.getLabelDict[recipe_type]
        annotations = items["annotations"]
        try:    #successful download from youtube
            #print("in _getSentence try")
            totalFrames = 0
            for annotation in annotations:
                #print("annotation=",annotation)
                segment=annotation["segment"]
                totalFrames += segment[1]-segment[0]+1
                #print("totalFrames=",totalFrames)
                #video = _getVideo(species=species, title=title, startidx=segment[0],endidx=segment[1])
                id = annotation["id"]
                sentence = annotation["sentence"]
                #print("id ={},sentence={}".format(id,sentence))
                rst.append([segment,sentence,id])
            #print("finish get annotations")
            while len(rst) >8:  #we only input 8 frames
                _min=1000
                _minidx = None
                for idx, r in enumerate(rst):
                    if (r[0][1]-r[0][0]) < _min:
                        _min = r[0][1]-r[0][1]
                        _minidx = idx
                totalFrames -= _min
                rst.pop(_minidx)

            #print("finish select 8 frmaes")
            #allocatie frame number
            #print("rst\n",rst)
            rst.sort(key=self._takeSeg, reverse=True)
            #print("after sort")
            #print(rst)
            allFrames = 8
            idx = 0
            #print(type(rst))
            ###rst = rst[:4]           #test bug 
            allocate = [0] * len(rst)
            #print("allocate")
            while allFrames >0:

                allocate[idx] +=1
                idx = (idx +1) % len(rst)
                allFrames -=1
            print(allocate)
            print("finish allocation numbers")
            for idx, r in enumerate(rst):
                spacing = (r[0][1]-r[0][0]) //(allocate[idx]+1)
                #print((r[0][1]-r[0][0]))
                #print(allocate[idx])
                #print("===")
                #print(spacing)
                frames = [r[0][0] + spacing*f for f in range(1,allocate[idx]+1) ]
                videos = []
                #print("in getting video")
                #print(frames)
                for frame in frames:
                    video = self._getVideo(species=species, title=title, startidx=frame,endidx=frame)
                    #print(video)
                    #print(video.size())
                    #exit()
                    print("success get vidoe clip")
                    videos.append(video)

                    ######do not comment it
                    print(len(videos)) ###use for check video return, if video return -1 will jump to exception
                    #### do not delete it

                    #exit()
                #print(videos)
                #exit()
                rst[idx].append(videos)
                #print(len(rst[idx][-1]))
                #exit()

            
            #print("rst\n",rst)
            rst.sort(key=self._takeid, )
            print("rst\n",rst)
            #print(rst)
            #exit()
            return rst
        except Exception as e:
            print(e)
            return -1


    def _getVideoTimeStamps(self,filename:str=None):
        #print("in _getVideoTimeStamps")
        #print(filename)

        if self.tmpFilename != filename:

            self.VideoTimeStamps = torchvision.io.read_video_timestamps(filename)
            self.tmpFilename = filename
        return self.VideoTimeStamps
        pass
    def _getVideo(self,species:str=None,title:str=None,startidx:int=None, endidx:int=None):    #get video clip in caption time
                            #return [video clip]
        title = ''.join(f for f in title if f.isalnum() or f == ' ').strip()
        filename = os.path.join(self.root_path,"images",species,title,self.ftmpl.format(startidx))
        #filename = r"/home/share/YouCook/downloadVideo/bibimbap/Bibimbap.mp4"      #resolution < input size
        #startidx = 1000
        #endidx = 1000
        print(filename,"is ",os.path.isfile(filename))

        #[ptslog, fps] = self._getVideoTimeStamps(filename=filename)


        #videoStream = torchvision.io.read_video(filename,ptslog[startidx],ptslog[endidx])
        videoStream = Image.open(filename)
        onlyVideo = videoStream

        #onlyVideo = onlyVideo.transpose(1,3)

        oriH, oriW = onlyVideo.size()[-2], onlyVideo.size()[-1]
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
            #print(np.shape(cropVideo))
            #return -1
        #print((oriH//2) +(min(oriH,oriW)//2))
        #print((oriH//2) -(min(oriH,oriW)//2))
        #print(onlyVideo.size())
        #exit()
        #onlyVideo = onlyVideo[...,(oriH-self.inputsize)//2:(oriH+self.inputsize)//2,(oriW-self.inputsize)//2:(oriW+self.inputsize)//2]
        #print(onlyVideo.size())
        #torchvision.transforms.ToTensor()
        #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(onlyVideo)
        
        #exit()
        #onlyVideo = self.transforms(onlyVideo)
        print("---after readvideo----")
        return torch.tensor(cropVideo)  #origin shape[frames,360,640,3] -->[frames, 3, 640, 360] -->[frames, 3, 224,224] and not normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        pass                
    def _getLabel(self):    #get cook type label, not id
        pass                #return recognition label
    def _getCaptions(self,species:str=None,title:str=None): #get video captions
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

            self.id2labelDict = dict((line[0],line[1]) for line in lines)
            labels = sorted(set(self.id2labelDict.keys()))
            self.getLabelDict = dict((label,idx) for idx, label in enumerate(labels))

        pass

    def __len__(self):  #for dataset iter need
        return len(self.llist)
    def __getitem__(self, index):   #for dataset iter need
                                    #return [label, URL, [[segment,sentence,id,videos], [], [], ...]
        #self.index = index
        item = self.llist[index].split("\n")[0]
        #print("item=",item)
        id = item.split("/")[0]    #100~130,200~230,300~325,400...
        URL = item.split("/")[1]   #aFsdfsd464
        #print("item=",item)
        #exit()
        try: #download successful
            #print(len(self.id2labelDict))
            #print(self.URL2titleDict)
            #print(self.getLabelDict)
            #exit()
            #print("id=",id)
            #print("URL=",URL)
            species = self.id2labelDict[id] #cooking type
            #print("species=",species)
            title = self.URL2titleDict[URL] #gordan steak
            #print("title=",title)
            label = self.getLabelDict[id] #predict label 0~100
            #print("label=",label)
            #exit()
            res = []
            res.append([label])
            res.append([URL])
            sentences = self._getSentence(URL=URL)
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


