#!/usr/bin/env python2
#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import glob
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os
import common

TRT_LOGGER = trt.Logger()


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw

def get_engine(onnx_file_path, max_batch_size, fp16_on, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_on
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
def download_file(path, link, checksum_reference=None):
    if not os.path.exists(path):
        print('downloading')
        wget.download(link, path)
        print()
    if checksum_reference is not None:
        raise ValueError('error')
    return path
def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    input_size = 416
    batch_size = 1
    fp16_on = True
    onnx_file_path = 'yolov3-tiny.onnx'
    engine_file_path = 'yolov3-tiny.trt'
    input_file_list = '/home/nvidia/yolov3-tiny2onnx2trt/imagelist.txt'
    IMAGE_PATH = '/home/nvidia/yolov3-tiny2onnx2trt/images/'
    save_path = '/home/nvidia/yolov3-tiny2onnx2trt/'
    
    output_shapes_416 = [(batch_size, 18, 13, 13), (batch_size, 18, 26, 26)]
    output_shapes_480 = [(batch_size, 18, 15, 15), (batch_size, 18, 30, 30)]
    output_shapes_544 = [(batch_size, 18, 17, 17), (batch_size, 18, 34, 34)]
    output_shapes_608 = [(batch_size, 18, 19, 19), (batch_size, 18, 38, 38)]
    output_shapes_dic = {'416': output_shapes_416, '480': output_shapes_480, '544': output_shapes_544, '608': output_shapes_608}
    
    with open(input_file_list, 'r') as f:
        filenames = []
        for line in f.readlines():
            filenames.append(line.strip())

    filenames = glob.glob(os.path.join(IMAGE_PATH, '*.jpg'))
    
    nums = len(filenames)
    # print(filenames)

    input_resolution_yolov3_HW = (input_size, input_size)
    
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    
    output_shapes = output_shapes_dic[str(input_size)]

    postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],
                          "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)],
                          "obj_threshold": 0.5, 
                          "nms_threshold": 0.35,
                          "yolo_input_resolution": input_resolution_yolov3_HW}

    postprocessor = PostprocessYOLO(**postprocessor_args)
    
    # Do inference with TensorRT
    filenames_batch = []
    images = []
    images_raw = []
    trt_outputs = []
    index = 0
    with get_engine(onnx_file_path, batch_size, fp16_on, engine_file_path) as engine, engine.create_execution_context() as context:
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        for filename in filenames:
            filenames_batch.append(filename)
            image_raw, image = preprocessor.process(filename)
            images_raw.append(image_raw)
            images.append(image)
            index += 1
            if index != nums and len(images_raw) != batch_size:
                continue
            inputs, outputs, bindings, stream = common.allocate_buffers(engine)
            images_batch = np.concatenate(images, axis=0)
            inputs[0].host = images_batch
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
            t2 = time.time()
            t_inf = t2 - t1
            print("time spent:",t_inf)
            print(len(trt_outputs))
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

	    print('test')
	    for i in range(len(filenames_batch)):
                fname = filenames_batch[i].split('/')
                fname = fname[-1].split('.')[0]
		img_raw = images_raw[i]
		shape_orig_WH = img_raw.size
		boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH), i)
		print("boxes size:",len(boxes))
		# Draw the bounding boxes onto the original input image and save it as a PNG file
		obj_detected_img = draw_bboxes(img_raw, boxes, scores, classes, ALL_CATEGORIES)
		output_image_path = save_path + fname + '_' + str(input_size) + '_bboxes.png'
		obj_detected_img.save(output_image_path, 'PNG')
		print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
            filenames_batch = []
            images_batch = []
	    images = []
	    images_raw = []
	    trt_outputs = []

if __name__ == '__main__':
    main()
