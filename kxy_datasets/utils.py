#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from urllib import request
import zipfile
import tarfile

def extract_from_url(url, dataset_path):
	'''
	Download a file locally and extract its content.
	'''
	assert url.endswith('.zip') or url.endswith('.tar.gz'), 'The file to unzip should be in the zip or tar.gz format'
	filename_zip = 'data.zip' if url.endswith('.zip') else 'data.tar.gz' if url.endswith('.tar.gz') else None
	file_path_zip = os.path.join(dataset_path, filename_zip)
	os.makedirs(dataset_path, exist_ok=True)
	if not os.path.exists(file_path_zip):
		request.urlretrieve(url, file_path_zip)
		if url.endswith('.zip'):
			zf = zipfile.ZipFile(file_path_zip)
			zf.extractall(dataset_path)

		if url.endswith('.tar.gz'):
			with tarfile.open(file_path_zip) as tf:
	def is_within_directory(directory, target):
		
		abs_directory = os.path.abspath(directory)
		abs_target = os.path.abspath(target)
	
		prefix = os.path.commonprefix([abs_directory, abs_target])
		
		return prefix == abs_directory
	
	def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
	
		for member in tar.getmembers():
			member_path = os.path.join(path, member.name)
			if not is_within_directory(path, member_path):
				raise Exception("Attempted Path Traversal in Tar File")
	
		tar.extractall(path, members, numeric_owner=numeric_owner) 
		
	
	safe_extract(tf, dataset_path)


