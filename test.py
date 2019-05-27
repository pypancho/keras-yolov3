import sys
import argparse
from yolo import YOLO
from PIL import Image
import cv2
import os
import glob
import numpy


full_path = '/home/smart3/yolo3_fuse/test'


def get_paths(full_path, image_type = '*.jpg'):
     
    total = 0 # The total numnber of images
    roots = []
    full_paths = []
    label_real_names = []
    
    for root, dirs, files in os.walk(full_path):
        roots.append(os.path.join(root, image_type))
        for dirc in dirs:
            label_real_names.append(dirc)
        for file in files:
            total = total + 1
    del roots[0]  # delete the root directory
    #print(roots)
    print("\nThe total number of full images is " + str(total) + ' .')
    print("The type name of full images are " + str(label_real_names))
         
       
    for i in range(0, len(roots)):
        full_paths.append(glob.glob(roots[i]))
    
    return sum(full_paths, [])

def detect_img(yolo):
    '''
    while True:
        
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            # r_image.show()
            opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite('pictures/test_result.png',opencvImage)
        '''
    #image = cv2.imread(img_path)
    full_paths = get_paths(full_path)
    for path in full_paths:
        print('Detecting:' + path.split('/')[-1])
        image = Image.open(path)
        r_image, no_boxes = yolo.detect_image([image,image])
        #r_image.show()
        if no_boxes!=0:
            opencvImage = cv2.cvtColor(numpy.array(r_image), cv2.COLOR_RGB2BGR)
            name = 'test_results/test_' + path.split('/')[-1]
            cv2.imwrite(name,opencvImage)



    '''
    image = Image.open(img_path)
    r_image, no_boxes = yolo.detect_image(image)
    #r_image.show()
    if no_boxes!=0:
        opencvImage = cv2.cvtColor(numpy.array(r_image), cv2.COLOR_RGB2BGR)
        name = 'test_results/test_' + img_path.split('/')[-1]
        cv2.imwrite(name,opencvImage)
    '''

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    FLAGS = parser.parse_args()
    print(vars(FLAGS))

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))