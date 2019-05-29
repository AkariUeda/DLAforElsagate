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
## Motion Vectors

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
### 3. Extracting the motion vectors

The extract_mvs tool need a list, for each movie, containing the frame numbers which it will extract the MPEG Motion Vectors.

Note: Since I-frames doesn't contain motion vectors, if one of the selected frames in the list is an I-frame, the motion vector from the next frame will be extracted instead.

This list can be generated from the list of 1fps current/previous frames. The `extract_motion_vectors.sh` script automatically creates this list for each video and extract the desired mvs'.

Run the line below replacing the necessary parameters:
```
./extract_motion_vectors.sh process_porn/porn_videos_list process_porn/porn_frames_previous porn_motion_vectors porn_mvs process_porn/porn
```

Where:

* porn_videos_list: list with your video absolute names (without path)
* porn_frames_previous: the previous frames list, generated during the frame extraction step
* porn_motion_vectors: directory to store auxiliary files
* porn_mvs directory: to store the extracted mvs
* process_porn/porn: path where to find the videos

### 4. Generate images from your motion vectors

```
generate_mvs_imgs.py -m mvs_dir/ -l lists/movies_name.list -o motions/ -os 224
```

Where:
* `mvs_dir` is the directory where you saved the mvs generated in the previous step
* movies_name.list is the name of the videos, without the full path.
* `motions` is the desired output directory where the script is going to save the motion vector images
* `224` is the dimension of the generated image. If you set it as 224, the images will be 224x224.