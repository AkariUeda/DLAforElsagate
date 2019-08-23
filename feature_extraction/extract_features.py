#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#  extract_features.py
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

'''
Extract features from the given images using caffe googlenet architecture.

- Authors: Mauricio Perez (mauriciolp84@gmail.com)
'''

import argparse
import os, string, subprocess, sys, os.path
from glob import glob
import numpy as np
import time
import caffe
from pathlib import Path
import multiprocessing
from joblib import Parallel, delayed
from skimage.transform import resize
import warnings

def parse_mean_binaryproto(filename):
	blob = caffe.proto.caffe_pb2.BlobProto()
	data = open( filename , 'rb' ).read()
	blob.ParseFromString(data)
	arr = np.array( caffe.io.blobproto_to_array(blob) )
	out = arr[0]
	out = np.ascontiguousarray(out.transpose(1,2,0))
	out = out.astype(np.uint8)

	return out

def load_args():
	ap = argparse.ArgumentParser(description='Extract features from the given images using caffe googlenet architecture.')

	ap.add_argument('-i', '--input_dir',
			dest='input_dir',
			help='path to the input directory, where the images are.',
			type=str, required=True)
	ap.add_argument('-l', '--imgs_list',
			dest='imgs_list',
			help='path to the list of imgs files to be processed.',
			type=str, required=True)
	ap.add_argument('-o', '--output_dir',
			help='path to the output directory',
			type=str, required=True)
	ap.add_argument('-p', '--proto_file',
			help='path to the prototxt file',
			type=str, required=True)
	ap.add_argument('-m', '--model_file',
			help='path to the model file, the pretrained net param',
			type=str, required=True)
	ap.add_argument('-ol', '--output_layer',
			help='name of layer to extract the features (expected a layer with flat output)',
			type=str, required=True)
	ap.add_argument('-a', '--mean_file',
			help='path to the mean file (expected a .binaryproto file)',
			type=str, required=False)
	ap.add_argument('-is', '--input_size',
			help='size of the input images [Width]x[Height] (Ex: 224x224)',
			type=str, required=False)
	ap.add_argument('-g', '--use_gpu', action='store_true',
			help='Use GPU, otherwise, use CPU',
			default = False, required=False)
	ap.add_argument('-gi', '--gpu_id',
			help='id of the gpu to use',
			type=int, default = 0, required = False)
	ap.add_argument('-ms', '--minibatch-size',
			help='Size of minibatch [100]',
			type=int, default = 100, required = False)
	ap.add_argument('-vt', '--verbose-time',
			help='Verbose time measures',
			action='store_true', required = False)
	ap.add_argument('-rw', '--rewrite',
			help='Rewrite existing files',
			default = False, required = False)

	args = ap.parse_args()

	return args

args = load_args()

# Creating output dir, if it doesnt exists already
if not os.path.exists(args.output_dir):
	os.makedirs(args.output_dir)

if not os.path.exists(args.imgs_list):
	print >> sys.stderr, "List of images doesn't exist"
	sys.exit(0)

##### Caffe options, transformer setting and network initiation BEGIN ####

if args.use_gpu:
	caffe.set_mode_gpu()
	caffe.set_device(args.gpu_id)
else:
	caffe.set_mode_cpu()

net = caffe.Net(args.proto_file, args.model_file, caffe.TEST)

net.blobs['data'].reshape(1,3,args.input_size.split('x')[0],args.input_size.split('x')[1])

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
if args.mean_file is not None:
	mean_mat = parse_mean_binaryproto(args.mean_file)
	transformer.set_mean('data', mean_mat.mean(0).mean(0)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

##### Caffe options, transformer setting and network initiation END ####

overall_input = []

imgs_files = open(args.imgs_list).read().split()
overall_input = []
input_files = [args.input_dir + '/' + filename for filename in imgs_files]

if args.rewrite == 'False':
	for arq in input_files:
		file_path = Path(arq)
		if file_path.exists() == False:
			overall_input.append(arq)
else:
	overall_input = input_files

input_size = (int(args.input_size.split('x')[0]),int(args.input_size.split('x')[1]))

nfiles = len(overall_input)
nbatches = np.ceil(nfiles / float(args.minibatch_size))

print 'A total of %d minibatches will be processed, containning at max %d images each, from a set of %d images overall' % (
		nbatches, args.minibatch_size, nfiles)

num_cores = multiprocessing.cpu_count()

for bidx, fidx in enumerate(xrange(0, nfiles, args.minibatch_size)):
	t0 = time.time()
	toprocess_input = overall_input[fidx:fidx+args.minibatch_size]

	def preprocess_img(im_f, transformer):
		#~ warnings.simplefilter('default')
		warnings.simplefilter('ignore')
		#~ warnings.simplefilter('always')
		image = caffe.io.load_image(im_f)
		image = resize(image, (args.input_size.split('x')[0], args.input_size.split('x')[1]),
                       anti_aliasing=True)
		print(image.shape)
		print("OIEEEE")
		"""
		Filter for suppressing the msg:
		/usr/local/lib/python2.7/dist-packages/scikit_image-0.11.3-py2.7-linux-x86_64.egg/skimage/external/tifffile/tifffile_local.py:3246: UserWarning: unexpected end of lzw stream (code 0)
		"""
		return transformer.preprocess('data', image)

	images_minibatch = np.array(Parallel(n_jobs=num_cores)(delayed(preprocess_img)(im_f, transformer) for im_f in toprocess_input))

	if args.verbose_time:
		t1 = time.time()
		print '\tReading data took %g seconds to process %d images' % (
			t1-t0, len(images_minibatch))

	net.blobs['data'].reshape(len(toprocess_input),3,input_size[0],input_size[1])

	net.blobs['data'].data[...] = images_minibatch

	net.forward()

	if args.verbose_time:
		t2 = time.time()
		print '\tnet.forward() took %g seconds to process %d images' % (
			t2-t1, len(images_minibatch))

	i = 0
	filename = ""
	f_size = 0

	### Process output, saving separately files ###
	for dsc_line in net.blobs[args.output_layer].data:
		if i >= len(toprocess_input):
			print  >> sys.stderr, "An Error has ocurred while processing the images"
			break

		rel_fname = os.path.relpath(toprocess_input[i], args.input_dir)
		output_fname = os.path.join(args.output_dir, rel_fname + ".dsc")
		output_dir = os.path.split(output_fname)[0]

		output_fname = output_fname.replace('\ ',' ')
		output_dir = output_dir.replace('\ ',' ')
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		i += 1

		line = '%d 1 1\n'%(dsc_line[:,0,0].shape[0])
		first = True
		for dsc_value in dsc_line[:,0,0]:
			if not first:
				line += ' '
			else:
				first = False
			line = line + '%.8f'%(dsc_value)

		f = open(output_fname, "w")
		f.write(line)

		f.close()

	if args.verbose_time:
		t3 = time.time()
		print '\tWriting data took %g seconds to process %d images' % (
			t3-t2, len(images_minibatch))

	print 'minibatch %d out of %d took %g seconds to process %d images' % (
		bidx+1, nbatches, time.time()-t0, len(toprocess_input))

print 'Extraction ended succesfully'
