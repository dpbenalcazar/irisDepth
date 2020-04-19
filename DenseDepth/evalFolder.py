import os
import glob
import time
import argparse
import PIL.Image as Image
import numpy as np
from matplotlib import pyplot as plt

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='irisDD_vanilla.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--inputs', default='iris/mini_LFVL/', type=str, help='Archivo o carpeta con imagenes de entrada')
parser.add_argument('--root_dir', default='', type=str, help='Directorio de origen la base de datos (para entradas de texto)')
parser.add_argument('--result_dir', default='results/mini_LFVL/', type=str, help='Directorio de salida')
parser.add_argument('--bs', default=6, type=int, help='Batch size.')
args = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

# Load model into GPU / CPU
print('\nLoading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)

# Load test data
print('\nLoading test data...')

inputImages = args.inputs # 'iris/irisT2Net_tes.txt' #'iris/Diego/' #'iris/mini_LFVL/' #'iris/Van-256x256_tes.txt'
root_dir = args.root_dir #'/home/danielb/DsIris3DCNN/' #'' #
result_dir = args.result_dir #'results/T2irisDepth/'#'results/Diego/' #'results/mini_LFVL/' #'results/irisDD_vanilla/'

# Check and create output folder
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

inp_img = []
opt_img = []
gt_img = []

if inputImages.find('.txt') != -1:
    text = open(inputImages,'rb').read()
    paths_AB = list((row.split('; ') for row in text.decode("utf-8").split('\n') if len(row) > 0))

    for path_A, path_B in paths_AB:
        path = os.path.join(root_dir , path_A.strip())
        inp_img.append(path)
        path = os.path.join(result_dir , path_A[-37:])
        opt_img.append(path)
        path = os.path.join(root_dir , path_B.strip())
        gt_img.append(path)

else:
    for root, _, fnames in os.walk(inputImages):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(inputImages , fname)
                inp_img.append(path)
                path = os.path.join(result_dir , fname)
                opt_img.append(path)

print('{} images loaded.\n'.format(len(inp_img)))

# Input image size:
im0 = Image.open(inp_img[0])
width, height = im0.size

# Make Predictions
print('Testing...')
start = time.time()

bs = args.bs
Nb = np.ceil(len(inp_img)/bs).astype('int32')

c = 0

for i in range(Nb):
    inputs = load_images(inp_img[i*bs:(i+1)*bs])
    outputs = predict(model, inputs, minDepth=0, maxDepth=255, batch_size=args.bs)
    for j in range(outputs.shape[0]): #bs
        pred = np.reshape(outputs[j],[192,192]) #[192,192]
        im = Image.fromarray(np.uint8(pred*255))
        im = im.resize([width, height])
        im.save(opt_img[c])
        c = c+1
    print('  - Batch {} of {}'.format(i+1,Nb))

end = time.time()
print('\nTest time {:0.1f} s'.format(end-start))
