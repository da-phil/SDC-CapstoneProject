import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import time
from glob import glob

cwd = os.path.dirname(os.path.realpath(__file__))


class TLClassifier(object):
    def __init__(self, threshold, hw_ratio, sim_testing, verbose=False):
        self.threshold = threshold
        self.hw_ratio = hw_ratio    #height_width ratio
        print("Initializing classifier with threshold={}, hw_ratio={}".format(threshold, hw_ratio))
        self.verbose = verbose
        self.tl_bbox = None

        self.num_pixels = 25
        # Define red pixels in hsv color space
        self.lower_red_1 = np.array([0,  70, 50],   dtype = "uint8")
        self.upper_red_1 = np.array([10, 255, 255], dtype = "uint8")

        self.lower_red_2 = np.array([170,  70,  50], dtype = "uint8")
        self.upper_red_2 = np.array([180, 255, 255], dtype = "uint8")

        self.category_index = [{"id": 0, "name": "red"},
                               {"id": 1, "name": "yellow"},
                               {"id": 2, "name": "green"},
                               {"id": 3, "name": "unknown"},
                               {"id": 4, "name": "unknown"}]  

        self.find_color_idx = lambda color: next((self.category_index[i]["id"]
                                            for i in range(len(self.category_index))
                                            if self.category_index[i]["name"] == color.lower()), 3)

        os.chdir(cwd)
        
        #if sim_testing, we use a detection and classification models
        #if site_testing, we use a single model which does both detection and classification
        
        if sim_testing: #we use different models for classification
            #tensorflow localization/detection model
            detect_model_name = "ssd_mobilenet_v1_coco_11_06_2017" #was "ssd_inception_v2_coco_11_06_2017"
            #detect_model_name = "ssd_mobilenet_v1_coco_28_01_2018"
            PATH_TO_CKPT = detect_model_name + "/frozen_inference_graph.pb"
            # setup tensorflow graph
            self.detection_graph = tf.Graph()
            # configuration for possible GPU
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # load frozen tensorflow detection model and initialize
            # the tensorflow graph
        
        else: # site testing where we load the localization+classification model        
            detect_model_name = "ssd_kyle_v4"
            PATH_TO_CKPT = detect_model_name + "/frozen_inference_graph.pb"
            # setup tensorflow graph
            self.detection_graph = tf.Graph()
        
            # configuration for possible GPU use
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # load frozen tensorflow detection model and initialize 
            # the tensorflow graph        
        
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name="")

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
              # Each bbox represents a part of the image where a particular object was detected.
            self.bboxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name("detection_scores:0")
            self.classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
            self.num_detections =self.detection_graph.get_tensor_by_name("num_detections:0")

    # Helper function to convert normalized bbox coordinates to pixels
    def bbox_normal_to_pixel(self, bbox, dim):
        height, width = dim[0], dim[1]
        bbox_pixel = [int(bbox[0]*height), int(bbox[1]*width), int(bbox[2]*height), int(bbox[3]*width)]
        return np.array(bbox_pixel)

    def get_localization(self, image, visual=False):
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            bounding box as list of integer coordinates [x_left, y_up, x_right, y_down]

        """

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (bboxes, scores, classes, num_detections) = self.sess.run(
                [self.bboxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            bboxes = np.squeeze(bboxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            cls = classes.tolist()
            #print("Classes: ", classes)

            # initial bounding box value
            bbox = [0, 0, 0, 0]

            # Find the first occurence of traffic light detection id=10
            idx = next((i for i, v in enumerate(cls) if v == 10), None)

            # If there is no detection
            if idx == None:
                if self.verbose:
                    print("No detection possible")

            # If the confidence of detection is too slow, 0.3 for simulator
            elif scores[idx] <= self.threshold:
                if self.verbose:
                    print("Detection has too low confidence: {}".format(scores[idx]))

            #If there is a detection and its confidence is high enough
            else:
                # checking orner cases
                dim = image.shape[0:2]
                bbox = self.bbox_normal_to_pixel(bboxes[idx], dim)
                bbox_h = bbox[2] - bbox[0]
                bbox_w = bbox[3] - bbox[1]
                ratio  = bbox_h / (bbox_w + 1e-3)
                
                # if the bbox is too small, 20 pixels for simulator
                if bbox_h < 20 or bbox_w < 20:
                    bbox = [0, 0, 0, 0]
                    if self.verbose:
                        print("Bounding box too small! ({}px)".format([bbox_h, bbox_w]))
                
                # if the bounding box height to width ratio is not right
                # approx. 1.5 for simulator, 0.5 for test site
                elif ratio < self.hw_ratio:
                    bbox =[0, 0, 0, 0]
                    if self.verbose:
                        print("Wrong h-w ratio ({})".format(ratio))
                else:
                    if self.verbose:
                        print("Localization confidence: {}".format(scores[idx]))

            self.tl_bbox = bbox

        return bbox


    def get_trafficlight_color(self, image):
        brightness_threshold    = 128

        red_vals, green_vals    = image[:,:,0].ravel(), image[:,:,1].ravel()
        red_pix_count           = np.where(red_vals >= brightness_threshold)[0].shape[0]
        green_pix_count         = np.where(green_vals >= brightness_threshold)[0].shape[0]
        red_pix_percentage      = 1.0*red_pix_count / red_vals.shape[0]
        green_pix_percentage    = 1.0*green_pix_count / green_vals.shape[0]

        brightness      = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,2] 
        hs, ws          = np.where(brightness >= (brightness.max()-50))
        tl_h            = image.shape[0]
        hs_mean         = hs.mean()
        hs_mean_norm    = hs_mean / tl_h

        # final classifier value is the vertical brightness distribution skewed up or down
        # by the percentage of red (minus -> upwards) and green (plus -> downwards) pixels
        classifier = hs_mean_norm - red_pix_percentage + green_pix_percentage        

        if self.verbose:
            print("Brightness mean:  {}".format(hs_mean_norm))
            print("Red pixels:       {}".format(red_pix_percentage))
            print("Green pixels:     {}".format(green_pix_percentage))
            print("Classifier value: {}".format(classifier))

        if classifier <= 0.45:
            signal_id = self.find_color_idx("red")
        elif classifier >= 0.60:
            signal_id = self.find_color_idx("green")
        else:
            signal_id = self.find_color_idx("yellow")

        return signal_id


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): cropped image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #img_resized = cv2.resize(image, (64, 64))
        img_full_np = np.asarray(image, dtype="uint8")

        start = time.time()
        bbox = self.get_localization(img_full_np)
        if self.verbose:
            print("Localization time: {}".format(time.time()-start))

        if self.verbose:
            print("Box:               {}".format(bbox))
        
        # If there is no detection or low-confidence detection
        if np.array_equal(bbox, np.zeros(4)):
            if self.verbose:
                print ("Could not localize traffic light")
            return self.find_color_idx("unknown"), bbox

        image_box = img_full_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        signal = self.get_trafficlight_color(image_box)
        return signal, bbox


if __name__ == "__main__":
    tl_cls = TLClassifier(0.1, 1.5, sim_testing=True, verbose=True)
    os.chdir(cwd)
    visual = True

    # problematic test images:
    #testimages = ["traffic_light_images/left0036.jpg"]

    # all test images:
    testimages  = glob(os.path.join("traffic_light_images/", "*.jpg"))
    for image_path in testimages:
        img_full    = Image.open(image_path)
        img_full_np = np.asarray(img_full, dtype="uint8")
        print("Processing following file: {}".format(image_path))

        start = time.time()
        signal, bbox = tl_cls.get_classification(img_full_np)
        print("Classification time: {}".format(time.time()-start))
        print("Detected light:      {}".format(tl_cls.category_index[signal]["name"]))

        if visual == True:
            # show bounding box image if box was detected and the whole image otherwise
            if np.array_equal(bbox, np.zeros(4)):
                image_vis = img_full_np
            else:
                image_vis = img_full_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            plt.figure(figsize=(6,4))
            plt.imshow(image_vis)
            plt.show()