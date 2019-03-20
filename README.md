
# Summary
* [ Database ](#database)
* [ Software Requirements ](#reqs)
* [ Data Preprocessing ](#preproc)
* [ Frame Extraction ](#frames)

<a name="database"></a>
# Database

The database can be downloaded from the [author's Google Drive](https://drive.google.com/open?id=12nWpZDxhQKC3c9N55F-azefqwgFg5PMl).

Note that you need permission to do so. Send an e-mail to `ueda.aka@gmail.com` so we can authorize your e-mail address to access the data.

In this link, you will find two directories with the training and validation set, in which you will find:
* The Elsagate videos downloaded from YouTube:  All videos from the **training set** were cut to have at maximum 3 minutes and 15 seconds (VERIFICAR). For the **validation set**, we used the full videos to validate the filtering method as it would work in production.
* The extracted and processed frames and motion vectors. The preprocessing steps are explained in the next section.
* The two folds in the training set used in the experiments.

<a name="reqs"></a>
# Software Requirements

Further explanations on when to use these library or other setup speficiations will be detailed on the next sections.

* Boost
* Caffe
* FFMPEG (Version used: 2.7.2)
* LIBSVM (Version used: 3.18) 
* OpenCV (Version used: 2.4.10)

# Methodology Pipeline

1. **Data Preprocessing**: Extract/Generate low-level data (static and/or motion)
2. Use a Deep Learning Architecture (DLA) model to extract the features from the low-level data
3. Pool the features into a single global  descriptor of the video
4. Predict the class of the video through frames and motion vectors separatedly using SVM
5. Fusion the frames and motion vectors scores to get a final classification

___

<a name="preproc"></a>
## Data Preprocessing

The data preprocessing pipeline was designed and proposed by Perez et al. in the paper [Video pornography detection through deep learning techniques and motion information](https://www.sciencedirect.com/science/article/pii/S0925231216314928). All the scripts used in the Data Preprocessing sections are under the copyright of Mauricio Perez.
___
<a name="frames"></a>
### Extract frames

#### 1. Requeriments:
* Boost

Compilation instructions for the tool frame_extractor:

1. Enter the frame_extractor dir
```
cd frame_extractor
```
2. Create a build dir
```
mkdir build; cd build
```
3. Compile 
```
cmake ..
make
```

#### 2. Extract frames

The following command extracts raw frames from the videos at a 1 frame per second rate.
Extract raw frames from the videos.

VERIFICAR A EXTRAÇÃO DE FRAMES PARA OS FLOWS

In fact, the tool `frame_extractor` not only extracts the frames associated to the frame sampling provided, but also extracts the next frame for each one of these frames, so it can be used for computing the Optical Flow Displacement Fields afterwards:

```
frame_extractor -t 12 -f 1 -p 100000 -i movies.list -o extracted_frames; * movies.list: List with the full path to the movies - Example of one line: "path/subpath/movie.mp4"
```


#### 3. Split Previous and Next lists
It's necessary to separate which frames are the selected frames sampling to be used and which are the next frames, only used for generating the Optical Flow representation. This can be done using `split_list_previous_next.bash` tool:
```
src/split_list_previous_next.bash frames.list .
```
Expected output:
```
frames_previous.list
frames_next.list
```
___
<a name="motions"></a>
### Extract Motion Vectors

#### 1. Requeriments:

* FFMPEG (Version used: 2.7.2)
* `pkg-config` installed or manual definition of enviroment variable `PKG_CONFIG_PATH` pointing to ffmpeg libs.
```
Example:

export PKG_CONFIG_PATH=~/sw/ffmpeg-2.7.2/build/lib/pkgconfig/
```
* `generate_mvs_imgs.py`
* `extract_mvs`

#### 2. Compilation instructions
1. Enter the `extract_mvs` dir
```
cd extract_mvs
```
2. Compile
```
make
```
___
## Feature Extraction


