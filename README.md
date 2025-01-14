Object detection uing Faster R-CNN
========

This repository implements [Faster R-CNN](https://arxiv.org/abs/1506.01497) with Resnet50 backbone. 
The aim was to create a simple implementation based on PyTorch faster r-cnn codebase and to get rid of all the abstractions and make the implementation easy to understand.

The implementation caters to batch size of 1 only and uses roi pooling on single scale feature map.
The repo is meant to train faster r-cnn on props and voc dataset. But can be customized for other datasets as well (see instructions).

On the PROPS dataset, we achieved 0.80 mAP and 0.67 mAP on the VOC 2007 dataset.

## Sample Output by training Faster R-CNN on PROPS dataset 
<img src="props/frcnn.gif" alt="Faster R-CNN" width="800" />

# Setup
* Create a new conda environment with python 3.10 and then install dependencies
* ```conda create -n fasterrcnn python=3.10.12```
* ```git clone https://github.com/spsingh37/object-detection.git```
* ```cd FasterRCNN-PyTorch```
* ```pip install -r requirements.txt```

<!-- * For training/inference use the below commands passing the desired configuration file as the config argument . 
* ```python -m tools.train``` for training Faster R-CNN on voc dataset
* ```python -m tools.infer --evaluate False --infer_samples True``` for generating inference predictions
* ```python -m tools.infer --evaluate True --infer_samples False``` for evaluating on test dataset -->

## Data preparation
For setting up the VOC 2007 dataset:
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007` folder
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007-test` folder

For setting up the PROPS dataset:
* Download the PROPS dataset from https://drive.google.com/file/d/1vG7_O-1JcYAgixdnV_n0QuFCt2R0050j/view?usp=share_link
* The PROPS dataset annotations are json format but we need them in xml format
* So execute the script 'json_to_xml.py' on the train.json and val.json (upon extracting the PROPS dataset)
* Place both the datasets inside the root folder of repo according to below structure
    ```
    FasterRCNN-Pytorch
        -> PROPS
            -> JPEGImages
            -> Annotations
        -> PROPS-test
            -> JPEGImages
            -> Annotations
        -> VOC2007
            -> JPEGImages
            -> Annotations
        -> VOC2007-test
            -> JPEGImages
            -> Annotations
        -> tools
            -> train_torchvision_frcnn.py
            -> infer_torchvision_frcnn.py
        -> config
            -> props.yaml
            -> voc.yaml
        -> model
            -> faster_rcnn.py
        -> dataset
            -> props.py
            -> voc.py
    ```

## For training on your own dataset

* Copy the PROPS config(`config/props.yaml`) and update the [dataset_params](https://github.com/spsingh37/object-detection/blob/main/config/props.yaml#L1) and change the [task_name](https://github.com/spsingh37/object-detection/blob/main/config/props.yaml#L35) as well as [ckpt_name](https://github.com/spsingh37/object-detection/blob/main/config/props.yaml#L41) based on your own dataset.
* Copy the PROPS dataset(`dataset/props.py`) class and make following changes:
   * Update the classes list [here](https://github.com/spsingh37/object-detection/blob/main/dataset/props.py#L61) (excluding background).
   * Modify the [load_images_and_anns](https://github.com/spsingh37/object-detection/blob/main/dataset/props.py#L13) method to returns a list of im_infos for all images, where each im_info is a dictionary with following keys:
     ```        
      im_info : {
		'filename' : <image path>
		'detections' : 
			[
				'label': <integer class label for this detection>, # assuming the same order as classes list present above, with background as zero index.
				'bbox' : list of x1,y1,x2,y2 for the bboxes.
			]
	    }
     ```
* Ensure that `__getitem__` returns the following:
  ```
  im_tensor(C x H x W) , 
  target{
        'bboxes': Number of Gts x 4,
        'labels': Number of Gts,
        }
  file_path(just used for debugging)
  ```
* Change the training script to use your dataset [here](https://github.com/spsingh37/object-detection/blob/main/tools/train_torchvision_frcnn.py#L41)
* Then run training with the desired config passed as argument.


## Using torchvision FasterRCNN 
* For training/inference using torchvision faster rcnn codebase, use the below commands passing the desired configuration file as the config argument.
* ```python -m tools.train_torchvision_frcnn``` for training using torchvision pretrained Faster R-CNN class on props dataset
   * This uses the following arguments other than config file
   * --use_resnet50_fpn
      * True(default) - Use pretrained torchvision faster rcnn
      * False - Build your own custom model using torchvision faster rcnn class)
* ```python -m tools.infer_torchvision_frcnn``` for inference and testing purposes. Pass the desired configuration file as the config argument.
   * This uses the following arguments other than config file
   * --use_resnet50_fpn
      * True(default) - Use pretrained torchvision faster rcnn
      * False - Build your own custom model using torchvision faster rcnn class)
      * Should be same value as used during training
   * --evaluate (Whether to evaluate mAP on test dataset or not, default value is False)
   * -- infer_samples (Whether to generate predicitons on some sample test images, default value is True)

## Configuration
* ```config/props.yaml``` - Allows you to play with different components of faster r-cnn on props dataset  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of FasterRCNN the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During inference the following output will be saved
* Sample prediction outputs for images in ```task_name/samples/*.png``` 

## Training plots for the PROPS dataset
<p align="center">
  <img src="props/rpn_classification_loss.png" alt="rpn_classification" width="400" />
  <img src="props/rpn_localization_loss.png" alt="rpn_localization" width="400" />
  <img src="props/frcnn_classification_loss.png" alt="frcnn_classification" width="400" />
  <img src="props/frcnn_localization_loss.png" alt="frcnn_localization" width="400" />
</p>
