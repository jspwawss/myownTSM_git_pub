# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pickle

import torch.utils.data as data
import torchvision
import torch

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import csv
import ast
import time
import multiprocessing as mp

count = 0
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 64


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def getword(self, word):
        return self.word2index[word]
    def getdict(self):
        return self.word2index

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    with open(filename,"rb") as dic:
        obj = pickle.load(dic)
        return obj

def capString(filepath:str):
    captions = str("")
    try:
        with open(filepath,"r") as captxt:
            #print("filepath=",filepath)
            lines = captxt.read().strip().split("\n\n")
            
            for line in lines:
                line = line.split("\n")
                #print(line)
                if isinstance(line[-1],list):   ##[Music] ..., not captions
                    #print(line)
                    continue
                    #exit()
                #if "music" in line[-1]:
                    #print(line[-1])
                    #exit()
                #    continue
                

                
                line[-1] = normalizeString(line[-1])
                captions += line[-1].strip() + str(" ")
        
            
    except Exception as e:
        print(e)     
        exit()     
    captions = captions.strip()
    #print(captions)
    #exit()
    return captions

def word2tensor(folderPath:str=os.path.join(r'/home/share/YouCook',"captions_v2")):
    #print("in word2tensor")
    engString=str("")
    for dirpath, dirnames, filenames in os.walk(folderPath):
        for filename in filenames:
            engString += capString(os.path.join(dirpath,filename)) + str(" ")
            #break
    #print("{0:*^50}".format("engstring"))
    engString = engString.strip()
    #print(len(engString))
    eng = Lang("english")
    eng.addSentence(engString)
    #print(eng.getdict)
    a = eng.getdict()

    save_object(a, "engDict.pkl")

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
    '''
    def __new__(cls, root_path = r'/home/share/YouCook',list_file=r"/home/share/YouCook",
                num_segments=8, new_length=1, modality="RGB",
                train=False,test=False,val=False,
                vtmpl = r"{}.mp4", ttmpl=r"{}.txt", ftmpl=r"image_{}.jpg",
                slice=False, recognition=False,
                transforms=torchvision.transforms.Compose([
                    torchvision.transforms.CenterCrop((256,256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    

                    ]),
                    inputsize = 224, hasWordIndex = False, hasPreprocess = False,
                ):
        print("in new")
        return object.__new__(cls)
    '''

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
                    inputsize = 224, hasWordIndex = False, hasPreprocess = False,
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

        self.hasPreprocess = hasPreprocess
        self.hasWordIndex = hasWordIndex
        if not self.hasWordIndex:
            #print("not hasWordIndex")
            word2tensor(folderPath=os.path.join(self.root_path,"captions_v2"))
            print("not hasWordIndex")
        print(" hasWordIndex")
        self.word2index = load_object("engDict.pkl")

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
        print("hasPreprocess" ,hasPreprocess)

        if not self.hasPreprocess:
            self._getAnnotation()
        else:
            self._getPreprocess(os.path.join(self.root_path,"preprocessDataset","dataset_"+self.mode+".pth"))
    def _getPreprocess(self, filename:str):
        print("in _get preprocess")
        self.llist = torch.load(filename)
        print(len(self.llist))

    def _getAnnotation(self):#only include train(1333) and val(457) = 1790
        with open(os.path.join(self.root_path,"annotations","youcookii_annotations_trainval.txt"),"r") as _anntxt:
            text = _anntxt.read()
            d = ast.literal_eval(text)
            self.database = d["database"]
            for key, value in self.database.items():

                if self.mode in value["subset"]:
                    self.llist.append([key,value["recipe_type"],value["annotations"],value["duration"]])

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
            if not os.path.isdir(os.path.join(self.root_path,"image_v2",str(id),URL)):
                return False

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
            return False

    def _getVideo(self,id:str=None, URL:str=None, idx:str=None):    #get video clip in caption time
                            #return [video clip]

        def _crop(self,i,j,onlyVideo, cropVideo ):
            onlyVideo = cropVideo
            out_queue.put(i,j)
            #return 
            pass
        #print("id={},url={}".format(type(id),type(URL)))
        filename = os.path.join(self.root_path,"image_v2",str(id),URL,self.ftmpl.format(str(idx)))
        #print("{0:*^50}".format("in getvideo"))
        #print(filename,"is ",os.path.isfile(filename))
        #return torch.tensor(np.zeros((1,3,256,256)))
        try:
            #s=time.time()
            videoStream = Image.open(filename)
            #e=time.time()
            #print("in video=",e-s)
        except:
            print("get video")
            return False
        self.onlyVideo = videoStream
        
        #onlyVideo = onlyVideo.transpose(1,3)
        self.onlyVideo = np.swapaxes(self.onlyVideo, 0,2)
        oriH, oriW = np.shape(self.onlyVideo)[-2], np.shape(self.onlyVideo)[-1]
        #oriH, oriW = onlyVideo.size()[-2], onlyVideo.size()[-1]
        m = min(oriH,oriW)
        a = oriH//2
        b = m//2
        c = oriW//2
        out_queue = mp.Queue()
        workers = []
        if m >= self.inputsize:
            #s=time.time()
            #print("ayayay")

            #onlyVideo = onlyVideo[...,(oriH//2-m//2) : (oriH//2+m//2), oriW//2-m//2:oriW//2+m//2]
            self.onlyVideo = self.onlyVideo[...,(a-b) : (a+b), c-b:c+b]
            #onlyVideo = onlyVideo[...,::min(oriH,oriW)/self.inputsize,::min(oriH,oriW)/self.inputsize]
            self.cropVideo = np.empty([1,3,self.inputsize,self.inputsize])
            for i in range(self.inputsize):
                for j in range(self.inputsize):
                    #print(i,j)
                    #workers.append(mp.Process(target=_crop, args=(i,j,self.cropVideo[...,i,j], self.onlyVideo[...,int(m/self.inputsize*i),int(m/self.inputsize*j)],\
                    #        out_queue)))
                    self.cropVideo[...,i,j] = self.onlyVideo[...,int(m/self.inputsize*i),int(m/self.inputsize*j)]
            #print(np.shape(cropVideo))
            #e=time.time()
            #print("video=",e-s)
        

        else:
            #print("ayayay")

            self.onlyVideo = self.onlyVideo[..., a-b : a+b, c-b:c+b]
            #onlyVideo = onlyVideo[...,::min(oriH,oriW)/self.inputsize,::min(oriH,oriW)/self.inputsize]
            self.cropVideo = np.empty([1,3,self.inputsize,self.inputsize])
            for i in range(self.inputsize):
                for j in range(self.inputsize):
                    self.cropVideo[...,i,j] = self.onlyVideo[...,int(min(oriH,oriW)/self.inputsize*i),int(min(oriH,oriW)/self.inputsize*j)]

        #print("---after readvideo----")
        
        return torch.tensor(self.cropVideo)  #origin shape[frames,360,640,3] -->[frames, 3, 640, 360] -->[frames, 3, 224,224] and not normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        

    def _getCaptions(self,id:str=None, URL:str=None, idx:str=None): #get video captions
                               #return [[captions, start time, endtime],[captions, start, end], ...]
        #print("{0:*^50}".format("in _getCaptions"))
        #print(os.path.join(self.root_path,"captions",species,self.ttmpl.format(title)),"is ",os.path.isfile(
        #    os.path.join(self.root_path,"captions",species,self.ttmpl.format(title))))

        captions = str("")
        try:
            
            with open(os.path.join(self.root_path,"captions_v6",id,self.ttmpl.format(URL)),"r") as captxt:
                lines = captxt.read().strip().split("\n\n")
                
                for line in lines:
                    #print("="*50)
                    line = line.split("\n")
                    if isinstance(line[-1],list):
                        #print(line[-1])
                        continue
                    #print(line)
                    #exit()
                    
                    #print(line)
                    start = line[1].split("-->")[0]
                    end = line[1].split("-->")[1]
                    sstart = start.split(",")[0].split(":")
                    eend = end.split(",")[0].split(":")

                    starttime = 0
                    endtime = 0 
                    for s,e in zip(sstart,eend):
                        starttime *=60
                        endtime *= 60
                        starttime += int(s)
                        endtime += int(e)
                    #print("starttime",starttime)
                    #print("endtime",endtime)
                    #print("idx",idx)
                    #print("type(line[-1])",type(line[-1]))
                    if int(idx)>= starttime:
                        #print("ayaya")
                        #print(line[-1])
                        if int(idx) <= endtime:
                            line[-1] = normalizeString(line[-1])
                            captions += line[-1].strip() + str(" ")
                        #print(captions)
                    else:
                        break
                    #print("{0:*^50}".format("captions"))
                    #print(captions)
                    
                    #exit()
        except Exception as e:
            print("get captions")
            print(captions)
            print(e)
        captions = captions.strip().split(" ")
        #print(captions)
        capRes = torch.zeros(MAX_LENGTH, dtype=torch.long,) 
        for idx, caption in enumerate(captions):
            capRes[idx] = torch.tensor(self.word2index[caption], dtype=torch.long,)

        #if len(captions) == 0:
        #    captions.append("") 
        #print(capRes)
        #print("capRes********")   
        if len(captions) == 1:
            if captions[0] == "":
                capRes[0] = torch.tensor(EOS_token, dtype=torch.long,   )
            else:
                capRes[len(captions)] = torch.tensor(EOS_token, dtype=torch.long)
        else:
            capRes[len(captions)] = torch.tensor(EOS_token, dtype=torch.long,   )
         
        #capRes = torch.tensor(capRes,dtype=torch.long,).view(-1,1)
        #print("capRes********") 
        #for c in capRes:
        #    if c == EOS_token:
                #print(c)
        #        break
        #    else:
        #        pass
        #print(capRes)
        #exit()
        return capRes, len(captions)

    def _label2Dict(self): #convert id to name
                            #and get dictionary from name to label
        with open(os.path.join(self.root_path,"label_foodtype.csv"),"r") as _csv:
        
            lines = csv.reader(_csv,delimiter=",")
            
            self.getLabelDict = dict((line[0],idx) for idx, line in enumerate(lines))

        pass

    def _getMode(self):
        return [self.mode, self.hasPreprocess, self.hasWordIndex]
    def __len__(self):  #for dataset iter need
        return len(self.llist)
    def __getitem__(self, index):   #for dataset iter need
                                    #return [URL, id, label, clip]
                                    #label = tensor([0,1,0,1,....])
                                    #clip = [[video, caption], [],...]
        #print("_gettiem")
        #print(self._getMode())
        #print(self.mode)
        #print(self.hasPreprocess)
        #exit()
        if not self.hasPreprocess:
            #self.index = index

            item = self.llist[index]
            #print(item)
            URL = item[0]   #aFsdfsd464
            id = item[1]    #100~130,200~230,300~325,400...
            rst = []
            annotation = item[2]
            #annotation = [{"segment": [35, 90], "id": 0, "sentence": "rinse the mussels"}, {"segment": [93, 124], "id": 1, "sentence": "slice the onions shallots and garlic"}, {"segment": [126, 146], "id": 2, "sentence": "fry the onions and garlic in a pan"}, {"segment": [164, 178], "id": 3, "sentence": "stir curry paste into the pan"}, {"segment": [202, 227], "id": 4, "sentence": "add fish sauce and coconut milk to the pan"}, {"segment": [275, 287], "id": 5, "sentence": "add the mussels to the sauce"}, {"segment": [294, 320], "id": 6, "sentence": "remove the stems from the cilantro"}, {"segment": [350, 364], "id": 7, "sentence": "add the cilantro to the mussels"}]

            #id = str("222")
            #URL = str("Bn0wtnfGUA8")
            #item[3] = 470.02
            
            #print(type(segment[0]))
            #exit()
            try: #download successful
                rst.append(URL)
                rst.append(id)
                #print(item[3])
                #print((int(item[3]))//10+1)
                #exit()
                if ((int(item[3]))//10+1) < 50:
                    l = (int(item[3]))//10+1
                else:
                    l = 50
                clip = [None] * (l+1)         #take one frame each 10 seconds

                #print(item[3])
                #print(len(clip))
                if  not os.path.isdir(os.path.join(self.root_path, "image_v2",str(id),URL)):
                    with open("imgDNE.txt","a+") as txt:
                        txt.write(str(id)+"\t"+str(URL)+"\n")
                    print("img dne")
                    return False
                for dirpath, dianames, filenames in os.walk(os.path.join(self.root_path, "image_v2",str(id),URL)):
                    
                    
                    for filename in filenames:
                        idx = filename.split(".")[0].split("_")[-1]
                        #print(idx)
                        if (int(idx) % (item[3]//(l-1))) != 0:
                            continue
                        #if int(idx) // 10 >=70:
                        #    continue
                        idx = int(idx) 
                        #s = time.time()
                        #print("{0:*^50}".format("idx "+str(idx)))
                        video = self._getVideo(id=id,URL=URL,idx=idx)
                        #print("after get video")
                        #e = time.time()
                        #print("video time = ",e-s)
                        if isinstance(video, bool) :
                            print("isinstance")
                            return False
                        #s = time.time()
                        captions, cap_num = self._getCaptions(id=id,URL=URL,idx=idx)
                        #e=time.time()
                        #print("caption time = ",e-s)
                        #print(captions)
                        #print(item[3])
                        #print(idx)
                        #print(len(clip))
                        #print(cap_num)
                        #print("after _getCaptions")
                        #print("idx = ",int(idx))
                        #print("l=",l)
                        #print("clip=",len(clip))
                        #print("item=",item[3])
                        #print(int(idx)//int(item[3]//(l-1)))
                        try:
                            clip[int(idx)//int(item[3]//(l-1))] = [video,captions]
                        except:
                            clip[-1] = [video, captions]
                    break
                    
                #print("-"*50)
                while clip[-1] is None:
                    clip.pop()
                label = torch.zeros(len(clip),dtype=torch.float)
                #print(label)
                #print("-"*50)
                for a in annotation:
                    #print(a)
                    segment = a["segment"]

                    for i in range(segment[0],segment[1]+1):
                        i = int(i) //(int(item[3]//(l-1)))
                        label[i] = True
                #print("label.size()",label.size())
                rst.append(label)
                rst.append(clip)
                #print("{0:*^50}".format("rst"))
                #print("dataset slice return len=",len(rst))


                #print("dataset slice return len=",len(rst))      
                #exit()
                return rst   

            except Exception as e:
                with open("dataerrrrrr.txt","a+") as txt:
                    txt.write(str(rst)+"\n")
                print("get item")
                print(e)
                #exit()
                #if index >2:

                    #exit()
                return False
        else:
            return self.llist[index]

        
        pass


if __name__ == "__main__":
    #word2tensor()
    #exit()
    #rst = load_object(os.path.join("/home/share/YouCook","preprocessDataset", "dataset_train.pth"))
    #rst = torch.load(os.path.join("/home/share/YouCook","preprocessDataset", "dataset_train_1.pth"))
    #print(rst[0])
    #print(len(rst))
    #exit()
    
    #word2index = load_object("engDict.pkl")
    #print(len(word2index))

    #exit()
    crop_size = 224
    dataset = YouCookDataSetRcg(val=True,inputsize=crop_size, hasWordIndex=True, hasPreprocess=False)
    dataset1 = YouCookDataSetRcg(train=True,inputsize=crop_size, hasWordIndex=False, hasPreprocess=False)
    #print(dataset._getMode())
    #print(dataset1._getMode())
    rst = []
    #print(dataset == dataset1)
    #exit()
    try:
        for i in dataset:
            #print(i)
            if isinstance(i, bool):
                print(i)
                continue
            print("ayaya")
            #print(i)
            
            [URL, id, label, clips] = i	
            for clip in clips:
                video = clip[0]
                caption = clip[1]
            rst.append(i)
            #save_object(rst, os.path.join("/home/share/YouCook","preprocessDataset", "dataset_train.pkl"))
            #break
            #print(i)
            #print("ayaya")
            #exit()
        print("test pass")
        torch.save(rst,os.path.join("/home/share/YouCook","preprocessDataset", "dataset_val.pth"))
        print("torch pass")
        #save_object(rst, os.path.join("/home/share/YouCook","preprocessDataset", "dataset_train.pkl"))
        #print("pickle pass")
    except Exception as e:
        
        print(i)
        print(e)

    