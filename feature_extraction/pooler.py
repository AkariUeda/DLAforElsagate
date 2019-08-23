#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  pooler.py
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

import sys
sys.path.append("../")

import argparse
from pathlib import Path
import time
import os
import numpy as np

from joblib import Parallel, delayed  
import multiprocessing

'''
Pool the features of the frames in the list (must be ordered), into a single global descriptor for each video.

- Authors: Mauricio Perez (mauriciolp84@gmail.com)
'''

def poolDscs(video,dsc_files, args):
	dsc_files = [ dsc for dsc in dsc_files if dsc.startswith(video) ]
	
	pooled_dsc = None

	#save all videos which the description extraction failed for some reason
	failed_videos = []	

	# Going through descriptions file
	for dsc_file in dsc_files:
		file_path = Path(args.frames_dsc_path+'/'+dsc_file+('' if dsc_file.endswith('.dsc') else '.dsc'))
		worked = False
		try:
			file_content = open(args.frames_dsc_path+'/'+dsc_file+('' if dsc_file.endswith('.dsc') else '.dsc')).read()			
			n_features = int(file_content.split('\n')[0].split(' ')[0])
			worked = True
		
		except:
			print('Problem with file {}'.format(file_path))
			video_name = '.'.join(dsc_file.split('.')[0:2])
			if video_name not in failed_videos:
				failed_videos.append(video_name)
			pass
		if worked:
			file_dsc = np.array(file_content.split('\n')[1].split(' ')[:n_features]).astype(float)

			# Possible point of bug: it expects that the file contains 'dx' only if it's the dx direction of the motion
			#	So, if the filename contains a 'dx' substring on it, this logic will fail
			if args.is_motion and 'dx' in dsc_file:
				complete_dsc = np.zeros((file_dsc.shape[0]*2))
				complete_dsc[:file_dsc.shape[0]] = file_dsc
			else:
				if args.is_motion:
					complete_dsc[file_dsc.shape[0]:] = file_dsc
				else:
					complete_dsc = file_dsc
		
				if pooled_dsc is None:
					pooled_dsc = complete_dsc
					n_frms = 1
				else:	# Performs pooling of actual description to the video dsc
					if args.pooling_type == 'max':
						pooled_dsc = np.maximum(pooled_dsc,complete_dsc)
					elif args.pooling_type == 'sum':
						pooled_dsc += complete_dsc
					elif args.pooling_type == 'avg':
						n_frms += 1
						pooled_dsc += complete_dsc



		
	# Saving last movie
	try:
		if args.pooling_type == 'avg':
			pooled_dsc = pooled_dsc/n_frms

		write_pooled_file(video, pooled_dsc, args.output_path, failed_videos)
	except:
		pass

	# for vid in failed_videos:
	# 	with open('failed_poolings.txt', 'a') as f:
	# 		print(vid)
	# 		f.write(vid+"\n")


def write_pooled_file(video, pooled_dsc, output_path, failed_videos):
	filename = output_path+'/'+video+'.dsc'
	if video not in failed_videos:
		np.savetxt(filename,\
			np.array([np.arange(np.shape(pooled_dsc)[0])+1,pooled_dsc]).transpose()\
			, fmt='%i:%.8f', newline=' ')
		
		try:
			file_pooled = open(filename,'r+')
			content = file_pooled.read()

			if video[:4] != 'safe':
				c = '1 '
			else:
				c = '-1 '
			
			content = c + content[:-1] + '\n'
			
			file_pooled.seek(0)
			file_pooled.truncate()

			file_pooled.write(content)
			file_pooled.close()
		except:
			print("Fail {}".format(filename))
			pass
# pooler class
class pooler():
	def __init__(self):
		# --- do the initial configuration ---------------------------------- #

		print '> Pooler -', time.asctime( time.localtime(time.time()) )

		args = self.load_args()
		
		print "Arguments:"
		print args

		self.build(args)

		print '\n> Pooler done -', time.asctime( time.localtime(time.time()) )

	def build(self, args):
		if not os.path.exists(args.output_path):
			os.makedirs(args.output_path)
		
		dsc_files = open(args.dsc_list_path).read().split()
		
		all_videos = set( [dsc_filename.split('.')[0] for dsc_filename in dsc_files] )
		videos = []
		if args.rewrite == True:
			for v in all_videos:
				try:
					filepath = args.output_path + '/' + v +'.dsc'
					print("Processando {}".format(filepath))
					if os.path.isfile(filepath) == False:
						print(v)
						videos.append(v)
				except:
					print("Fail: {}".format(v))
					pass
		else:
			videos = all_videos
			
		num_cores = multiprocessing.cpu_count()

		print "Processing a total of",len(videos),"videos and", len(dsc_files),"frames."
		print "Using",num_cores,"cores."
		
		Parallel(n_jobs=num_cores, verbose=10)(delayed(poolDscs)(video,dsc_files, args) for video in videos)

	def progress(self,max,done):
		if (100*done/max)%5 == 0:
			i = 100*done/max
			sys.stdout.write('\r')
			sys.stdout.write("[%-20s] %d%%" % ('='*(i/5), i))
			sys.stdout.flush()

	# load arguments from stdin
	def load_args(self):
		ap = argparse.ArgumentParser(description='Pool overfeat features from each video of the porn Data.')

		ap.add_argument('-d', '--frames-descriptions-file-path',
						dest='frames_dsc_path',
						help='Path to the frames descriptions files.',
						type=str, required=True)
		ap.add_argument('-l', '--frames-descriptions-file-list',
						dest='dsc_list_path',
						help='Path to the list of frames description files.',
						type=str, required=True)
		ap.add_argument('-o', '--output-path',
						dest='output_path',
						help='Path to output the pooled descriptions file.',
						type=str, required=True)
		ap.add_argument('-p', '--pooling-type',
						help='Pooling type ([avg]; max; sum)',
						default='avg',
						type=str, required=False)
		ap.add_argument('-m', '--is-motion', dest='is_motion',
						help='Process as motion fetures. Will concatenate dx and dy.',
						default=False,
						action='store_true', required=False)
		ap.add_argument('-f', '--filename', dest='logfile_name',
						help='Name of the error log file.',
						default='pooling.log',
						type=str, required=False)
		ap.add_argument('-r', '--rewrite', dest='rewrite',
						help='Rewrite already existing pooled descriptions.',
						default=False, type=bool,
						required=False)
		return ap.parse_args()

p = pooler()
