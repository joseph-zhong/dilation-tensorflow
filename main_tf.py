#!/usr/bin/env python3
import tensorflow as tf
import pickle
import cv2
import os
import os.path as path
from utils import predict, predict_no_tiles
from model import dilation_model_pretrained
from datasets import CONFIG


if __name__ == '__main__':
    test = True
    # Choose between 'cityscapes' and 'camvid'
    dataset = 'cityscapes'

    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Image in / out parameters
    input_image_path  = path.join('data', dataset + '.png')
    output_image_path = path.join('data', dataset + '_out.png')

    # Build pretrained model and save it as TF checkpoint
    with tf.Session() as sess:

        # Choose input shape according to dataset characteristics
        if not test:
            input_h, input_w, input_c = CONFIG[dataset]['input_shape']
        else:
            input_h, input_w, input_c = (1452, 2292, 3) # REVIEW: dr-eye-ve size.
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))
        asdf = saver.save(sess, path.join(checkpoint_dir, 'dilation.ckpt'))
        print("saved asdf:", asdf)

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        output = 'softmax:0'
        model = graph.get_tensor_by_name(output)
        model = tf.reshape(model, shape=(1,)+CONFIG[dataset]['output_shape'])

        # Read and predict on a test image
        input_image = cv2.imread(input_image_path)
        # import matplotlib.pyplot as plt
        # plt.imshow(input_image)
        # plt.show()
        input_tensor = graph.get_tensor_by_name('input_placeholder:0')
        if test:
            tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
            for tensor in tensors:
                print(tensor)
            import numpy as np
            import os

            path = '/home/josephz/tmp/data/dr-eyeve/35/frames/0057.png'
            image = cv2.imread(path)

            # output = 'input_placeholder:0'
            outputs = ('conv1_1/Relu:0', 'conv1_2/Relu:0',
                       # 'pool1/MaxPool:0',
                       'conv2_1/Relu:0', 'conv2_2/Relu:0',
                       # 'conv3_1/Relu:0', 'conv3_2/Relu:0', 'conv3_3/Relu:0',
                       'conv5_3/Relu:0',
                       'fc6/Relu:0',
                       'fc7/Relu:0',
                       'final/Relu:0',
                       'ctx_pad1_1:0',
                       'ctx_conv1_1/Relu:0',
                       'ctx_conv7_1/Relu:0',
                       'ctx_fc1/Relu:0',
                       'ctx_final/BiasAdd:0',
                       'ctx_upsample/Relu:0',
            )
            for output in outputs:
                print("Checking", output)
                import pdb
                pdb.set_trace()
                model = graph.get_tensor_by_name(output)
                outp = os.path.join('/home/josephz/ws/git/ml/framework/scripts/dilation/outs/tf', output.split('/')[0])
                if not os.path.isfile(outp + '.npy'):
                    print("Saving to ", outp)
                    y = predict_no_tiles(image, input_tensor, model, dataset, sess, test=test)
                    np.save(outp, y)

            out_tensor = graph.get_tensor_by_name('softmax:0')
            out_tensor = tf.reshape(out_tensor, shape=(1,) + (1080, 1920, 19))
            y = predict_no_tiles(image, input_tensor, out_tensor, dataset, sess, test=False)
        else:
            # Convert colorspace (palette is in RGB) and save prediction result
            predicted_image = predict(input_image, input_tensor, model, dataset, sess, test=test)
            predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_image_path, predicted_image)



