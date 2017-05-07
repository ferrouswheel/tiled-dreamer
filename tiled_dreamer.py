import argparse
import time
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

def load_model():
    # If your GPU supports CUDA and Caffe was built with CUDA support,
    # uncomment the following to run Caffe operations on the GPU.
    caffe.set_mode_gpu()
    caffe.set_device(0) # select GPU device if multiple devices exist

    model_path = '../caffe/models/bvlc_googlenet/' # substitute your path here
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data


def src_coords(i, overlap, window_inner):
    h = window_inner
    b = overlap
    x1 = i * (b + h)
    x2 = (i + 1) * (b + h) + b
    return x1, x2


def dest_coords(x1, x2, max_x, window_outer, b):
    hb = b / 2 # half border
    if x1 == 0:
        xd1 = 0
        xc1 = 0
    else:
        xd1 = x1 + hb
        xc1 = hb

    if x2 == max_x:
        xd2 = x2
        xc2 = window_outer
    else:
        xd2 = x2 - hb
        xc2 = window_outer - hb
    return xd1, xd2, xc1, xc2


def make_step_split(net, octave_base, detail, max_dim=(320, 320), step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''
    img = octave_base+detail
    
    #print img.shape, max_dim
    vsplit = (img.shape[1] / max_dim[0]) + 1;
    hsplit = (img.shape[2] / max_dim[1]) + 1;
    
    #print "splits", vsplit, hsplit

    overlap = 32

    h = (img.shape[1] - ((vsplit + 1) * overlap)) / vsplit
    H = img.shape[1]
    w = (img.shape[2] - ((hsplit + 1) * overlap)) / hsplit
    W = img.shape[2]
    window_h = h + (2 * overlap)
    window_w = w + (2 * overlap)

    #print "img.shape", img.shape
    #print "overlap", overlap
    #print "h", h, window_h
    #print "w", w, window_w

    #print vsplit * (overlap + h) + overlap
    #print hsplit * (overlap + w) + overlap

    new_H = (vsplit * (overlap + h) + overlap)
    new_W = (hsplit * (overlap + w) + overlap)

    if new_H != H or new_W != W:
        #print "Cropping input image to match windows and overlaps"
        img = img[0:new_H, 0:new_W, :]
        H = new_H
        W = new_W

    result = np.zeros(img.shape)

    src = net.blobs['data']
    dst = net.blobs[end]
    src.reshape(1,3,window_h,window_w) # resize the network's input image size
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    
    b = overlap
    for i in range(0,vsplit):
        for j in range(0,hsplit):
            x1, x2 = src_coords(i, b, h)
            y1, y2 = src_coords(j, b, w)

            subwindow = img[:, x1:x2, y1:y2]

            #print "subwindow shape", subwindow.shape
            #clear_output(wait=True)
            #subdream = deepdream(net, subwindow, iter_n=1)
            
            src.data[0] = subwindow
            
            src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            net.forward(end=end)
            objective(dst)  # specify the optimization objective
            net.backward(start=end)
            g = src.diff[0]
            # apply normalized ascent step to the input image
            src.data[:] += step_size/np.abs(g).mean() * g

            src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

            if clip:
                bias = net.transformer.mean['data']
                src.data[:] = np.clip(src.data, -bias, 255-bias)

            xd1, xd2, xc1, xc2 = dest_coords(x1, x2, H, window_h, b)
            yd1, yd2, yc1, yc2 = dest_coords(y1, y2, W, window_w, b)

            #print x1, x2, "=>", xd1, xd2
            #print y1, y2, "=>", yd1, yd2

            subdream = src.data[0] - octave_base[:, x1:x2, y1:y2]
            result[:, xd1:xd2, yd1:yd2] = subdream[:, xc1:xc2, yc1:yc2]
    return result
    

def deepdream_split(net, base_img, max_dim=(575, 1024), iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, save_steps=None, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)
        
        for i in xrange(iter_n):
            start = time.time()
            detail = make_step_split(net, octave_base, detail, max_dim, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, octave_base + detail)
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            if save_steps:
                PIL.Image.fromarray(np.uint8(vis)).save(save_steps)
            print octave, i, end, vis.shape, time.time() - start, 'seconds'
        octave_result = octave_base + detail
    # returning the resulting image
    return deprocess(net, octave_result)


if __name__ == "__main__":
    import sys
    import os
    from random import shuffle

    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list-layers', action='store_true')
    parser.add_argument('-t','--target-layer', action='store', default='inception_4c/output')
    parser.add_argument('--iters', action='store', type=int, default=15)
    parser.add_argument('--octaves', action='store', type=int, default=5)
    parser.add_argument('--explore', action='store_true')
    parser.add_argument('-i','--in-file', action='store')
    parser.add_argument('-o','--out-file', action='store', default='out.jpg')

    args = parser.parse_args()
    net = load_model()

    if args.list_layers:
        print net.blobs.keys()
        sys.exit(0)

    big_img = np.float32(PIL.Image.open(args.in_file))

    if args.explore:
        base, ext = os.path.splitext(args.out_file)
        # skip the early layers and the very end layers
        layers_to_explore = net.blobs.keys()[8:-4]
        shuffle(layers_to_explore)
        for l in layers_to_explore:
            if 'split' in l:
                continue
            fn = base + '_' + l.replace('/', '__') + ext
            if os.path.exists(fn):
                print l, "output file", fn, "already exists"
                continue
            print "========= LAYER", l, "=========="
            start = time.time()
            dream_bug=deepdream_split(net, big_img, max_dim=(640, 640), save_steps=fn, iter_n=args.iters, octave_n=args.octaves, end=l)
            PIL.Image.fromarray(np.uint8(dream_bug)).save(fn)
            print "Total time:", time.time() - start
    else:
        dream_bug=deepdream_split(net, big_img, max_dim=(640, 640), save_steps=args.out_file, iter_n=args.iters, octave_n=args.octaves, end=args.target_layer)
        PIL.Image.fromarray(np.uint8(dream_bug)).save(args.out_file)