#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  generate_mvs_imgs.py
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

import time
import numpy as np
import argparse, sys, os
from cv2 import normalize,cvtColor,imwrite,NORM_MINMAX,resize

from joblib import Parallel, delayed  
import multiprocessing

def load_args():
	ap = argparse.ArgumentParser(description='Generates images from Motion Vector information extracted using "extract_mvs". Wiil generate images to ALL of the frames which contains motion information on the file.')

	ap.add_argument('-m', '--mvs-file-path',
					dest='mvs_path',
					help='path to the mvs files.',
					type=str, required=True)
	ap.add_argument('-l', '--videos-list',
					dest='videos_list_path',
					help='path to the list of videos.',
					type=str, required=True)
	ap.add_argument('-o', '--output-path',
					dest='output_path',
					help='path to output the extracted frames.',
					type=str, required=True)
	ap.add_argument('-os', '--out-shape',
					dest='out_shape',
					help='Center crop and resize the generated images to the passed value. If this \
						parameter is not present, generated images will have original size.',
					type=int, required=False)

	args = ap.parse_args()
	
	print args
	
	return args

def rescale(img, new_size=(224,224)):
	# Will perform a resize having the smaller size as the smaller new_size and cropped at the center
	
	#~ width, height, channels = img.shape
	width, height = img.shape[0], img.shape[1]
	new_width, new_height = new_size

	### Resizing
	aspect_ratio = 1.*height / width
	
	if (height > width):
		dsize = (int(np.ceil(height * (1.0 * new_width/ width))), new_width)
	else:
		dsize = (new_height, int(np.ceil(width * (1.0 * new_height/ height))))

	rescaled_img = resize(img,dsize)
	
	dsize_h, dsize_w = dsize
	
	### Croping at center
	left = (dsize_w - new_width)/2
	top = (dsize_h - new_height)/2
	right = (dsize_w + new_width)/2
	bottom = (dsize_h + new_height)/2
	rescaled_img = rescaled_img[left:right, top:bottom]
	
	return rescaled_img

def read_mvs(filename):
	_file = open(filename)
	mvs_values = _file.read().split('\n')[1:-1]
	if len(mvs_values) <= 1:
		print("ERRO: mvs esta vazio: "+ filename)	
	mvs = []
	#~ framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags
	#~ 2,-1,16,16,   8,   8,   8,   8,0x0
	for line in mvs_values:
		line_values = line.split(',')
		mvs += [{ 
			'framenum': int(line_values[0]), 
			'source': int(line_values[1]),
			'blockw': int(line_values[2]),
			'blockh': int(line_values[3]),
			'srcx': int(line_values[4]),
			'srcy': int(line_values[5]),
			'dstx': int(line_values[6]),
			'dsty': int(line_values[7]),
			'flags': line_values[8]
		}]
		
	_file.close()
	
	return mvs
	
def compute_motion(mvs):
	# TODO: Caution with the size estimation! *Test*
	# Using last dst position and block size
	#~ size = (mvs[-1]['dstx']+mvs[-1]['blockw']/2,mvs[-1]['dsty']+mvs[-1]['blockh']/2)
	size = (mvs[-1]['dsty']+mvs[-1]['blockh']/2,mvs[-1]['dstx']+mvs[-1]['blockw']/2)
	
	# TODO: Using src size as output size?
	# size = (XXX,XXX)
	
	motion_dx = np.zeros(size)
	motion_dy = np.zeros(size)
	
	for motion in mvs:
		if motion['source'] < 0: 
			diff_dx = motion['dstx'] - motion['srcx']
			diff_dy = motion['dsty'] - motion['srcy']
		elif motion['source'] > 0:
			diff_dx = motion['srcx'] - motion['dstx']
			diff_dy = motion['srcy'] - motion['dsty']
		else:
			diff_dx = motion['dstx'] - motion['srcx']
			diff_dy = motion['dsty'] - motion['srcy']
			print "WARNING: source is not 1 nor -1"
		
		x_begin = motion['dstx'] - motion['blockw']/2
		x_end = motion['dstx'] + motion['blockw']/2 
		y_begin = motion['dsty'] - motion['blockh']/2
		y_end = motion['dsty'] + motion['blockh']/2 
		
		motion_dx[y_begin:y_end, x_begin:x_end] = diff_dx
		motion_dy[y_begin:y_end, x_begin:x_end] = diff_dy
	
	return motion_dx, motion_dy

def progress(max,done):
	if (100*done/max)%5 == 0:
		i = 100*done/max
		sys.stdout.write('\r')
		sys.stdout.write("[%-20s] %d%%" % ('='*(i/5), i))
		sys.stdout.flush()

def processVideo(video, args):
	mvs = read_mvs(args.mvs_path + '/' + video + '.mvs')
		
	frames_list = np.unique([motion['framenum'] for motion in mvs])

	for curr_frame in frames_list:
		mvs_frame = [motion for motion in mvs if motion['framenum']==curr_frame]
		
		#~ print curr_frame, mvs_frame[0]['source']
		
		motion_dx, motion_dy = compute_motion(mvs_frame)
		
		dx = np.zeros_like(motion_dx)
		dy = np.zeros_like(motion_dy)
		
		dx[:,:] = normalize(motion_dx,None,0,255,NORM_MINMAX)
		dy[:,:] = normalize(motion_dy,None,0,255,NORM_MINMAX)
		#~ 
		#~ rgb_dx = cvtColor(gray_dx_meanSub,COLOR_GRAY2BGR)
		#~ rgb_dy = cvtColor(gray_dy_meanSub,COLOR_GRAY2BGR)
		
		if args.out_shape is not None:
			dx = rescale(dx,(args.out_shape,args.out_shape))
			dy = rescale(dy,(args.out_shape,args.out_shape))
		
		curr_frame_string = '{0:07d}'.format(curr_frame)
		
		imwrite(args.output_path + '/' + video + '-' + curr_frame_string + '_dx.tif',dx)
		imwrite(args.output_path + '/' + video + '-' + curr_frame_string + '_dy.tif',dy)

def main():
	args = load_args()
	
	print '> Generate mvs images -', time.asctime( time.localtime(time.time()) )
	
	videos = open(args.videos_list_path).read().split()
	
	if not os.path.isdir(args.output_path):
		os.makedirs(args.output_path)
	
	print "Processing a total of",np.shape(videos)[0],"videos."
	
	num_cores = multiprocessing.cpu_count()

	print "Using",num_cores,"cores."
	
	Parallel(n_jobs=num_cores)(delayed(processVideo)(video, args) for video in videos)
	
	print '\n> Generate mvs images done -', time.asctime( time.localtime(time.time()) )
	
	return 0

if __name__ == '__main__':
	main()

