<a name="preproc"></a>
# Data Preprocessing

The data preprocessing pipeline was designed and proposed by Perez et al. in the paper [Video pornography detection through deep learning techniques and motion information](https://www.sciencedirect.com/science/article/pii/S0925231216314928). All the scripts used in the Data Preprocessing sections are under the copyright of Perez et al..

<a name="frames"></a>
## Extract Frames

### 1. Requeriments:
* Boost
* `./data_preprocessing/frame_extractor` script.

You can use the compiled binary that can be found in `/build` directory, or you can recompile it using the following instructions: 

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

### 2. Extract Frames

The following command extracts raw frames from the videos at a 1 frame per second rate.


```
frame_extractor -t 12 -f 1 -p 100000 -i movies.list -o extracted_frames; * movies.list: List with the full path to the movies - Example of one line: "path/subpath/movie.mp4"
```
**P.S:** In fact, the tool `frame_extractor` not only extracts the frames associated to the frame sampling provided, but also extracts the next frame for each one of these frames, so it can be used for computing Optical Flow Displacement Fields, another motion information feature we studied in our project.


### 3. Split Previous and Next Lists
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
## Extract Motion Vectors

### 1. Requeriments:

* FFMPEG (version 2.7.2)
* `./data_preprocessing/generate_mvs_imgs.py`
* `./data_preprocessing/extract_mvs`: You can use the already compiled binary or recompile it using the instructions in the next section.
* `pkg-config` installed or manual definition of enviroment variable `PKG_CONFIG_PATH` pointing to ffmpeg libs.
```
Example:
export PKG_CONFIG_PATH=~/sw/ffmpeg-2.7.2/build/lib/pkgconfig/
```
### 2. Compilation Instructions
1. Enter the `extract_mvs` dir
```
cd extract_mvs
```
2. Compile
```
make
```

