#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from urllib import request
import zipfile

def extract_from_url(url, dataset_path):
	'''
	Download a file locally and extract its content.
	'''
	filename_zip = 'data.zip'
	file_path_zip = os.path.join(dataset_path, filename_zip)
	os.makedirs(dataset_path, exist_ok=True)
	if not os.path.exists(file_path_zip):
		request.urlretrieve(url, file_path_zip)
		zf = zipfile.ZipFile(file_path_zip)
		zf.extractall(dataset_path)

