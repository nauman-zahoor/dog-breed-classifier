from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt  

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm

from keras.applications.resnet50 import preprocess_input, decode_predictions

from models.extract_bottleneck_features import * ############ Need this file

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


# defining models
# extract pre-trained face detector
print('Setting up Face Detection Model...')
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml') ############ Need this file

# define ResNet50 model
print('Setting up Dog Detection Model...')
ResNet50_model = ResNet50(weights='imagenet')

# define InceptionV3 model
print('Setting up Dog Breed Identification Model...')
InceptionV3_model = Sequential()
InceptionV3_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
InceptionV3_model.add(Dense(133, activation='softmax'))
InceptionV3_model.load_weights('./models/weights.best.InceptionV3.hdf5')

# laod dog_breeds 
dog_names = np.load('./models/dog_names.npz')   ############ Need this file
dog_names = dog_names['arr_0']


# Functions

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0




def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):  # using resenet for dog detection
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

# return predicted dog breed of image
def InceptionV3_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = InceptionV3_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
    
    
    
# algo to check dog breed
def detect_dog_breed(img_path):
    '''
    Take input image
    Check if human or dog or none
    if none:
        say no human nor dog found
    elif human:
        run dog breed detector and say human resembles this dog breed
    else:
        run dog breed detector and print dog breed
        
    INPUT:
        img_path: path to input image
        
    OUTPUT:
        dog_breed: name of the dog breed or NONE
        
    '''
    
     
    # detect if human or dog is present in image
    human_present = face_detector(img_path)>0
    dog_present = dog_detector(img_path)
    
    dog_breed = ''
    human_or_dog = ''
    if human_present:
        dog_breed = InceptionV3_predict_breed(img_path)
        #print('Human in image looks like they belongs to following dog breed : ',dog_breed)
        human_or_dog = 'HUMAN'
        
    elif dog_present:
        dog_breed = InceptionV3_predict_breed(img_path)
        #print('Dog in image belongs to following breed: ',dog_breed)
        human_or_dog = 'DOG'
    
    else:
        #print('No Human or Dogs were Detected...')
        dog_breed = 'Unknown'
        human_or_dog = 'Unknown'

    return human_or_dog, str(dog_breed)
        
     
        
if __name__ == '__main__':
    # testing humans
    img = './test_images/human2.jpg'
    print('testing image:',img)
    human_or_dog, dog_breed = detect_dog_breed(img)

    if human_or_dog == 'Unknown' and dog_breed == 'Unknown':
        print('No Human or Dog found in image...')
    else:
        print(human_or_dog , ' found in image belonging to',dog_breed, ' breed!' )
        
    

    img = './test_images/goat.jpg'
    print('testing image:',img)
    human_or_dog, dog_breed = detect_dog_breed(img)
    if human_or_dog == 'Unknown'  and dog_breed  == 'Unknown' :
        print('No Human or Dog found in image...')
    else:
        print(human_or_dog , ' found in image belonging to',dog_breed, ' breed!' )