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
                vtmpl = r"{}.mp4", ttmpl=r"{}.txt",
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
        
        filename = os.path.join(self.root_path,"downloadVideo",species,self.vtmpl.format(title))
        #filename = r"/home/share/YouCook/downloadVideo/bibimbap/Bibimbap.mp4"      #resolution < input size
        #startidx = 1000
        #endidx = 1000
        print(filename,"is ",os.path.isfile(filename))

        [ptslog, fps] = self._getVideoTimeStamps(filename=filename)
        #print("ptslog=",len(ptslog))
        #print("fps=",fps)

        videoStream = torchvision.io.read_video(filename,ptslog[startidx],ptslog[endidx])
        onlyVideo = videoStream[0]
        #print("*"*50)
        onlyVideo = onlyVideo.transpose(1,3)
        #print(onlyVideo.size())
        #print(type(onlyVideo))
        #print(onlyVideo)
        #for i in onlyVideo:
        #    print(i)
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
            #captions = self._getCaptions(species=species, title=title)
            #print(type(captions))
            #print("{0:*^50}".format("after captions"))
            #print(np.shape(captions))
            #print(captions)
            #for caption in captions:
            #    print(caption)
            #    [cap, start, end] = caption
            #    video = self._getVideo(species=species, title=title,start=start,end=end)
            #    print("{0:*^50}".format("after _getVideo"))
            #    #yield from [label, video, cap]
            #    
            #    res.append([label, video, cap])
            #    print("after appending")
            return res                      
        except Exception as e:
            print(e)

            #if index >2:

            #    exit()
            return False

        
        pass


class YouCookDataSet(data.Dataset):
    def __init__(self, root_path,list_file,
                num_segments=8, new_length=1, modality="RGB",
                train=False,test=False,
                vtmpl = r"{}.mp4", ttmpl=r"{}.txt",
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
        self.vtmpl = vtmpl
        self.ttmpl = ttmpl
        self.tmpFilename = None
        self.VideoTimeStamps = None
        if self.test ^ self.train == False:
            print("neither train nor tets")
            return False
        self._getList(mode="train" if self.train else "test")
        self._URL2title()
        self._label2Dict()

    def _getList(self,mode:str=None):
        with open(os.path.join(self.root_path,"split",mode+"_list.txt")) as listtxt:
            self.llist = listtxt.readlines()
        
    def _URL2title(self):   #convert URL 2 title name
        with open(os.path.join(self.root_path,"url2label.txt"),"r") as _txt:
            lines = _txt.readlines()
            #print("lines=",lines)
            self.URL2titleDict = dict((line.split("\t")[0].strip(),line.split("\t")[1].strip()) for line in lines)
        pass
    def _getSentence(self): #get annotations sentence
        pass                #return ?????, still thinking

    def _getVideoTimeStamps(self,filename:str=None):
        print("in _getVideoTimeStamps")
        print(filename)

        if self.tmpFilename == filename:
            print("?>?")
            return self.VideoTimeStamps
        else:
            print("non timestamps")
            self.VideoTimeStamps = torchvision.io.read_video_timestamps(filename)
            self.tmpFilename = filename
            return self.VideoTimeStamps
        pass
    def _getVideo(self,species:str=None,title:str=None,start:str="::", end:str="::"):    #get video clip in caption time
                            #return [video clip]
        
        filename = os.path.join(self.root_path,"downloadVideo",species,self.vtmpl.format(title))
        print(filename,"is ",os.path.isfile(filename))
        #tmp =  torchvision.io.read_video_timestamps(filename)
        #[ptslog,fps] = torchvision.io.read_video_timestamps(filename)
        [ptslog, fps] = self._getVideoTimeStamps(filename=filename)
        print("ptslog=",len(ptslog))
        print("fps=",fps)
        #print("{0:*^50}".format("after torch io read video timestamps"))
        startlog, endlog = start.replace(",",".").split(":"), end.replace(",",".").split(":")
        startsec, endsec = float(0),float(0)
        #print(startlog)
        for s in startlog:
            startsec = startsec*60 + float(s)
            #print("startsec = ",startsec)
        for e in endlog:
            endsec = endsec*60 + float(e)
        print("----cal pts----")
        startpts = float(startsec) / float(fps)
        endpts = float(endsec) / float(fps) 
        print("startlog={},endlog={}".format(startlog,endlog))
        print("startpts={},endpts={}".format(startpts,endpts))
        videoStream = torchvision.io.read_video(filename,ptslog[int(startpts)],ptslog[int(endpts)])
        print("---after readvideo----")
        return videoStream  #origin shape[frames,360,640,3]
        pass                
    def _getLabel(self):    #get cook type label, not id
        pass                #return recognition label
    def _getCaptions(self,species:str=None,title:str=None): #get video captions
                               #return [[captions, start time, endtime],[captions, start, end], ...]
        print("in _getCaptions")
        print(os.path.join(self.root_path,"captions",species,self.ttmpl.format(title)),"is ",os.path.isfile(
            os.path.join(self.root_path,"captions",species,self.ttmpl.format(title))))
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
                                    #return [[label, video clip, captions], [], [], ...]
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
            print(self.getLabelDict)
            #exit()
            print("id=",id)
            print("URL=",URL)
            species = self.id2labelDict[id] #cooking type
            print("species=",species)
            title = self.URL2titleDict[URL] #gordan steak
            print("title=",title)
            label = self.getLabelDict[id] #predict label 0~100
            print("label=",label)
            #exit()
            res = []
            captions = self._getCaptions(species=species, title=title)
            print(type(captions))
            print("{0:*^50}".format("after captions"))
            print(np.shape(captions))
            #print(captions)
            for caption in captions:
                print(caption)
                [cap, start, end] = caption
                video = self._getVideo(species=species, title=title,start=start,end=end)
                print("{0:*^50}".format("after _getVideo"))
                #yield from [label, video, cap]
                
                res.append([label, video, cap])
                print("after appending")
            return res                      
        except:
            if index >2:

                exit()
            return False

        
        pass


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False, data_fuse = False,wholeFrame=False):
        print("in dataset.py, __init__")
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.data_fuse = data_fuse      #if modality is RGB, fuse RGB and depth; else if modality is depth, fuse depth and skeleton
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.wholeFrame = wholeFrame        #try to base on Wu's 
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality in ['RGB', 'RGBDiff']:
            #print("in dataset.py 63")
            try:
                if self.data_fuse:
                    #print("in dataset.py")
                    #print("self.root_path, directory, self.image_tmpl.format(idx)) = {},{},{}".format(self.root_path, directory, self.image_tmpl.format(idx)))
                    #exit()
                    rgb_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')
                    depth_img = Image.open(os.path.join(self.root_path, directory.replace('sk_color', 'skeleton'), self.image_tmpl.format(idx))).convert('L')
                    return [rgb_img, depth_img]
                else:
                    #print("in dataset.py")
                    #print("self.root_path, directory, self.image_tmpl.format(idx)) = {},{},{}".format(self.root_path, directory, self.image_tmpl.format(idx)))
                    #exit()
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                #print("in dataset.py")
                #print("self.root_path, directory, self.image_tmpl.format(idx)) = {},{},{}".format(self.root_path, directory, self.image_tmpl.format(idx)))
                #exit()
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx+1))).convert('RGB')]
            #exit()
        elif self.modality == 'Depth':
            try:
                if self.data_fuse:
                    depth_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('L')
                    skeleton_img = Image.open(os.path.join(self.root_path, directory.replace('sk_depth', 'skeleton'), self.image_tmpl.format(idx))).convert('L')
                    return [depth_img, skeleton_img]
                else:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('L')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx+1))).convert('L')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx+1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        #print("tmp=",tmp)
        #exit()
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))
        #print("video_list=",self.video_list)
        #exit()

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])
            return offsets + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
        #if True:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            #print(np.array(offsets)+1)
            #exit()
            return np.array(offsets) + 1
        elif self.twice_sample:
            #print("num_frames=",record.num_frames)
            #print("self.new_length=",self.new_length)
            #print("self.num_segments,",self.num_segments)
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            #print("tick=",tick)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])
            #offsets = np.arange(record.num_frames)
            #print("offset=",offsets)

            return offsets + 1
        elif self.wholeFrame:
            start,end = 1,9
            offsets = []
            while end <= record.num_frames+1:
                #print(record.num_frames)
                offsets.append([x for x in range(start,end)])
                #print(offsets)
                start +=1
                end = start + 8
            #print("dataset py 217")
            return [x for x in range(1,(end-1)//8*8+1)]
            #return self.getWholeFrame(offsets=offsets)
            #exit()

        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1
    def getWholeFrame(self,offsets=None,frameNum=0):
        print("in getWhoeFrame")
        print("frameNum = ",frameNum)
        print("len=",len(offsets))
        while frameNum < len(offsets):
            print("in getWholeFrame")
            print(offsets[frameNum])
            yield offsets[frameNum]
            frameNum+=1
        frameNum = 0

    def __getitem__(self, index):
        #print("in dataset.py, __getitem__")
        #print("index=",index)
        record = self.video_list[index]
        #print("record=",record)
        # check this is a legit video folder

        #print("self.imgae_tmpl=",self.image_tmpl)
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        #print("filename=",file_name)
        #print("full_path=,",full_path)
        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        #print("**"*50)
        #print(segment_indices)
        #print("*"*50)
        return self.get(record, segment_indices)
    
    def get(self, record, indices):
        #print("in dataset.py, get")
        #print("indices,",indices)
        images = list()
        #print("record.num_frames=",record.num_frames)
        for seg_ind in indices:
            #print("seg_ind=",seg_ind)
            p = int(seg_ind)
            #exit()
            for i in range(self.new_length):
                #print("i=",i)
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                   
                    p += 1
            #print("p",p)
        #print(len(images))
        process_data = self.transform(images)
        #print("process data isze",process_data.size())
        return process_data, record.label
    '''
    def get(self, record, indices):
        #print("in dataset.py, get")
        #print("indices,",indices)
        images = list()
        for indice in indices:
            for seg_ind in indice:
                p = int(seg_ind)
                #print("seg_ind=",seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

            process_data = self.transform(images)
            return process_data, record.label
    '''

    def __len__(self):
        #print("in dataset.py __len__")
        return len(self.video_list)
