import os
import imageio
from PIL import Image
import shutil
from tensorflow.keras.preprocessing.image import load_img


CONTENT_IMG_PATH = './content/august.jpg'
STYLE_IMG_PATH = './style/wave-art.jpg'

cImg = load_img(CONTENT_IMG_PATH)
sImg = load_img(STYLE_IMG_PATH)

contentName = CONTENT_IMG_PATH[CONTENT_IMG_PATH.rfind('/') + 1 : CONTENT_IMG_PATH.rfind('.')]
styleName = STYLE_IMG_PATH[STYLE_IMG_PATH.rfind('/') + 1 : STYLE_IMG_PATH.rfind('.')]

# Make new directory to save outputs to
path = os.path.join('./test', '%s-%s' % (styleName, contentName))
os.mkdir(path)
SAVE_PATH = '%s/%s-%s.jpg' % (str(path), styleName, contentName)

shutil.copyfile(CONTENT_IMG_PATH, '%s/%s.jpg' % (str(path), contentName))
shutil.copyfile(STYLE_IMG_PATH, '%s/%s.jpg' % (str(path), styleName))
imageio.imwrite(SAVE_PATH, cImg)
