# Guideline

## 0. Environment Preparation
- python>=3.6
- apex
- pytorch=1.5.0+cu101
- torchvision=0.6.0+cu101
- tensorboardX
- h5py
- opencv
- scikit-learn
- yacs
 
## 1.Prepare datasets
### 1.1  Download dataset :
-  UCF-Crime: [Download](https://www.crcv.ucf.edu/projects/real-world/)
-  ShanghaiTech: [Download](https://svip-lab.github.io/dataset/campus_dataset.html)

### 1.2.Transform the datasets into h5py form

For training speed, we translate the video datasets into a single h5py file for reducing the indexing time in Disk
By keeping the compressed type as JPG, we can reduce the memory space.

Here, we give the example as translating UCF-Crime training set into a single h5py file by conducting code in `utils/make_h5.py`

If you do not want to make h5 file, you can modify the `datasets/dataset.py` to adapt the usage of raw videos.

### 1.3 Downlaod pretrained weights
MIST finetuned checkpoints are uploaded on [OneDrive](https://1drv.ms/u/s!Ai48CHyipiNUkFTHTQGze7QLY1Fn?e=lhkr0i)

## 2. Path Modification
All the default paths are set in `configs/constant.py`. You should the file as the paths set in the file, or modify it.

## 3. Testing
You can test the model via the command as below:
```shell script
python testing/test.py --gpus 0,1,2,3 --MODEL SHT_C3D
```

The argument `--MODEL` is to choose the model structure and pretrained weights, 
which should be one of the candidate lists `[SHT_C3D, SHT_I3D, UCF_C3D, UCF_I3D]`

Specifically, if you wanna to make some visualization, you should choose `UCF_C3D` and use command below:
```shell script
python testing/test.py --gpus 0,1,2,3 --MODEL UCF_C3D --vis_UCF
```