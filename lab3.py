
import os, sys, getopt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
# from scipy.misc import imsave, imresize
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model 
import warnings
import datetime

tf.compat.v1.disable_eager_execution()
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = ""           #TODO: Add this.
STYLE_IMG_PATH = ""             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.0003125    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1e-5

TRANSFER_ROUNDS = 3

SAVE_PATH = None

globalLoss = None
globalGrads = None
getLossAndGradients = None
#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape((3, CONTENT_IMG_H, CONTENT_IMG_W))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def getGradients(x):
    global globalLoss, globalGrads
    assert globalLoss is not None
    grads = np.copy(globalGrads)
    globalLoss = None
    globalGrads = None
    return grads


def getOutput(layers, model):
    output = list()
    for layer in layers:
        output.append(model.get_layer(layer).output)
    print(output)
    return output


#========================<Loss Function Builder Functions>======================

def styleLoss(styles, gStyles):
    N = 3
    M = CONTENT_IMG_H * CONTENT_IMG_W
    return K.sum(K.square(gramMatrix(gStyles) - gramMatrix(styles))) / (4. * (N ** 2) * (M ** 2))
    # styleLoss = K.variable(0.)
    # w = 0.2
    # for style, gStyle in zip(styles, gStyles):  
    #     M_l = CONTENT_IMG_H * CONTENT_IMG_W
    #     N_l = K.int_shape(gStyle)[0]
    #     # print('(%s, %s)' % (N_l, M_l))
    #     loss = w * (K.sum(K.square(gramMatrix(gStyle) - gramMatrix(style))) / (4. * (N_l ** 2) * (M_l ** 2)))
    #     loss = K.print_tensor(loss)
    #     styleLoss.assign_add(loss)
    # # print('styleLoss:', styleLoss)
    # return styleLoss


def contentLoss(content, gContent):
    return K.sum(K.square(gContent - content))


def totalLoss(x):
    a = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] -
                 x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] -
                 x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def calculateLoss(x):
    global globalLoss, globalGrads, getLossAndGradients
    assert globalLoss is None
    x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    outs = getLossAndGradients([x])
    loss = outs[0]
    grads = outs[1].flatten().astype('float64')
    globalLoss = loss
    globalGrads = grads
    return loss



#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    # tImg = np.random.randint(256, size=(CONTENT_IMG_H, CONTENT_IMG_W, 3)).astype('float64')
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #img = imresize(img, (ih, iw, 3))
        img.resize(ih, iw, 3)
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.constant(cData) # variable?
    styleTensor = K.constant(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)

    model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputTensor)

    outputDict = dict([(layer.name, layer.output) for layer in model.layers])

    print("   VGG19 model loaded.")
    loss = K.variable(0.)
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    gContentOutput = contentLayer[2, :, :, :]
    loss = loss + (CONTENT_WEIGHT * contentLoss(contentOutput, gContentOutput))

    print("   Calculating style loss.")
    # styleModels = [Model(inputs=model.input, outputs=model.get_layer(layer).output) for layer in styleLayerNames]
    # styleOutputs = [outputDict[layerName][1, :, :, :] for layerName in styleLayerNames]
    # gStyleOutputs = [outputDict[layerName][2, :, :, :] for layerName in styleLayerNames]
    
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        gStyleOutput = styleLayer[2, :, :, :]

        loss = loss + ((STYLE_WEIGHT / len(styleLayerNames)) * styleLoss(styleOutput, gStyleOutput))   #TODO: implement.
    loss = loss + (TOTAL_WEIGHT * totalLoss(genTensor))

    # TODO: Setup gradients or use K.gradients().
    global getLossAndGradients
    grads = K.gradients(loss, genTensor)[0]
    assert grads is not None
    getLossAndGradients = K.function([genTensor], [loss, grads])

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        x, tLoss, _ = fmin_l_bfgs_b(func=calculateLoss,
                                    x0=tData.flatten(),
                                    fprime=getGradients,
                                    maxfun = 20)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x)
        # saveFile = None   #TODO: Implement.
        # imsave(SAVE_PATH, img)   #Uncomment when everything is working right.
        img.save('%s-%s' % (SAVE_PATH, i))
        print("      Image saved to \"%s\"." % SAVE_PATH)
    print("   Transfer complete.")



#=========================<Main>================================================

def parseArgs():
    global CONTENT_IMG_PATH, STYLE_IMG_PATH, SAVE_PATH, CONTENT_WEIGHT, STYLE_WEIGHT
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, 's:c:a:b:h')
    except:
        raise ValueError('Unrecognized argument. See -h for help')

    for opt, arg in opts:
        if opt == '-c':
            STYLE_IMG_PATH = arg
        elif opt == '-s':
            CONTENT_IMG_PATH = arg
        elif opt == 'a':
            if (arg < 0 or arg > 1):
                raise ValueError('Content weight must be in [0, 1]')
            CONTENT_WEIGHT = float(arg)
        elif opt == 'b':
            if (arg < 0 or arg > 1):
                raise ValueError('Style weight must be in [0, 1]')
            STYLE_WEIGHT = float(arg)
        elif opt == '-h':
            print('Usage: \n\
                -c <path to content image>\n\
                -s <path to style image>\n\
                -a <content weight>\n\
                -b <style weight>\n')
            sys.exit()

    contentName = CONTENT_IMG_PATH[CONTENT_IMG_PATH.rfind('/') + 1 : CONTENT_IMG_PATH.rfind('.')]
    styleName = STYLE_IMG_PATH[STYLE_IMG_PATH.rfind('/') + 1 : STYLE_IMG_PATH.rfind('.')]
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    SAVE_PATH = './output/%s-%s' % (styleName, contentName)



def main():
    parseArgs()
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    # tData = np.random.randint(256, size=(CONTENT_IMG_H, CONTENT_IMG_W, 3)).astype('float32')
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
