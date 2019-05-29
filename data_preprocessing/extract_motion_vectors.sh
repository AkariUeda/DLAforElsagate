#!/bin/sh

#extract_motion_vectors.sh, to execute, run the line below with the following parameters:
#porn_videos_list list with your video absolute names (without path)
#porn_frames_previous the previous frames list, generated during the frame extraction step
#porn_motion_vectors directory to store auxiliary files
#porn_mvs directory to store the extracted mvs
#process_porn/porn path where to find the videos
#
#./extract_motion_vectors.sh process_porn/porn_videos_list process_porn/porn_frames_previous porn_motion_vectors porn_mvs process_porn/porn

list=$1
previouslist=$2
folder=$3
mvs=$4
source=$5

mkdir -p $folder
mkdir -p $mvs
echo $list
cat $list | while read video
do
  echo "Processing: $video"
  grep $previouslist -e $video | cut -d'-' -f2 > temp.list;
  total_frames=$(wc -l temp.list | cut -d' ' -f1);
  echo $total_frames > $folder/$video.list;
  cat temp.list >> $folder/$video.list;
  rm -f temp.list
  /home/akari/dataset/scripts/extract_mvs $source/$video $folder/$video.list > $mvs/$video.mvs;
done


