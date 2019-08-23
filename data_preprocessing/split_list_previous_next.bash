#!/bin/bash
# -*- coding: utf-8 -*-
#
#  split_list_previous_next.bash
#  
#  Copyright 2016 Mauricio Perez <mperez@mperez>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

#set -e

## "set -e" is for exiting on first error encountered

###
#
#	Split frames list in previous and next list
#		
#
###

############ Parameters ##############
list=$1
out_dir=$2
group=$3

echo "BEGIN - Split list -"`date`

echo "Parameters:"
echo "list: "$list
echo "out_dir: "$out_dir
echo "group: "$group

if [ $# -ne 2 ]
  then
    echo "Missing or extra arguments."
    echo "Exiting."
    #exit 0
fi

############ Main ##############

videos=$(cut -d'.' -f-2 < $list | sort -u)

rm -f $out_dir/"$group"_frames_previous $out_dir/"$group"_frames_next

for video in $videos; do
	echo -en '\t'$video'        '
	grep $list -e $video | sed -n 1~2p >> $out_dir/"$group"_frames_previous
	grep $list -e $video | sed -n 2~2p >> $out_dir/"$group"_frames_next
done;

echo "END - Split list -"`date`

read;
