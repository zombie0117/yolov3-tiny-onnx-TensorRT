# yolov3-tiny2onnx2trt
convert your yolov3-tiny model to trt model

#device: nvidia tx2


#jetpack version:jetpack4.2:
	ubuntu18.04 
	tensorrt5.0.6.3 
	cuda10.0 
	cudnn7.3.1


#others:
	python=2.7
	numpy=1.16.1
	onnx=1.4.1 (important)
	pycuda=2019.1.1
	Pillow=6.1.0
	wget=3.2


# custom settings

	data_processing.py:
		line14: LABEL_FILE_PATH = '/home/nvidia/yolov3-tiny2onnx2trt/coco_labels.txt'
		line19: CATEGORY_NUM = 80

	yolov3_to_onnx.py:
		line778: img_size = 416
		line784: cfg_file_path = '/home/nvidia/yolov3-tiny2onnx2trt/yolov3-tiny.cfg'
		line811: weights_file_path = '/home/nvidia/yolov3-tiny2onnx2trt/yolov3-tiny.weights'
		line826: output_file_path = 'yolov3-tiny.onnx'

	onnx_to_tensorrt.py:
		line39: input_size = 416
		line40: batch_size = 1
		line42~line46:
		    onnx_file_path = 'yolov3-tiny.onnx'
		    engine_file_path = 'yolov3-tiny.trt'
		    input_file_list = '/home/nvidia/yolov3-tiny2onnx2trt/imagelist.txt'
		    IMAGE_PATH = '/home/nvidia/yolov3-tiny2onnx2trt/images/'
		    save_path = '/home/nvidia/yolov3-tiny2onnx2trt/'

#steps:
1.change your settings as "#custom settings"

2.
	cd yolov3-tiny2onnx2trt
	python yolov3_to_onnx.py
	
	you will get a yolov3-tiny.onnx file

3.	
  python onnx_to_tensorrt.py:
	
	you will get a yolov3-tiny.trt file and some inferenced images.

