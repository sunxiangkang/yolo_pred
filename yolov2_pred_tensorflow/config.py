import numpy as np


modelPath=r'.\weights\checkpoint_dir'

trainedModelWeightsNames=[
    ('conv1/kernel', [3, 3, 3, 32]),('conv1_bn/gamma', [32]),('conv1_bn/beta', [32]),('conv1_bn/moving_mean', [32]),('conv1_bn/moving_variance', [32]),
    ('conv2/kernel', [3, 3, 32, 64]),('conv2_bn/gamma', [64]),('conv2_bn/beta', [64]),('conv2_bn/moving_mean', [64]),('conv2_bn/moving_variance', [64]),
    ('conv3_1/kernel', [3, 3, 64, 128]),('conv3_1_bn/gamma', [128]),('conv3_1_bn/beta', [128]),('conv3_1_bn/moving_mean', [128]),('conv3_1_bn/moving_variance', [128]),
    ('conv3_2/kernel', [1, 1, 128, 64]),('conv3_2_bn/gamma', [64]),('conv3_2_bn/beta', [64]),('conv3_2_bn/moving_mean', [64]),('conv3_2_bn/moving_variance', [64]),
    ('conv3_3/kernel', [3, 3, 64, 128]),('conv3_3_bn/gamma', [128]),('conv3_3_bn/beta', [128]),('conv3_3_bn/moving_mean', [128]),('conv3_3_bn/moving_variance', [128]),
    ('conv4_1/kernel', [3, 3, 128, 256]),('conv4_1_bn/gamma', [256]),('conv4_1_bn/beta', [256]),('conv4_1_bn/moving_mean', [256]),('conv4_1_bn/moving_variance', [256]),
    ('conv4_2/kernel', [1, 1, 256, 128]),('conv4_2_bn/gamma', [128]),('conv4_2_bn/beta', [128]), ('conv4_2_bn/moving_mean', [128]),('conv4_2_bn/moving_variance',[128]),
    ('conv4_3/kernel', [3, 3, 128, 256]),('conv4_3_bn/gamma', [256]), ('conv4_3_bn/beta', [256]),('conv4_3_bn/moving_mean', [256]),('conv4_3_bn/moving_variance', [256]),
    ('conv5_1/kernel', [3, 3, 256, 512]),('conv5_1_bn/gamma', [512]),('conv5_1_bn/beta', [512]),('conv5_1_bn/moving_mean',[512]), ('conv5_1_bn/moving_variance', [512]),
    ('conv5_2/kernel', [1, 1, 512, 256]),('conv5_2_bn/gamma', [256]), ('conv5_2_bn/beta', [256]),('conv5_2_bn/moving_mean', [256]),('conv5_2_bn/moving_variance', [256]),
    ('conv5_3/kernel', [3, 3, 256, 512]),('conv5_3_bn/gamma', [512]),('conv5_3_bn/beta', [512]),('conv5_3_bn/moving_mean', [512]),('conv5_3_bn/moving_variance', [512]),
    ('conv5_4/kernel', [1, 1, 512, 256]),('conv5_4_bn/gamma', [256]),('conv5_4_bn/beta', [256]),('conv5_4_bn/moving_mean', [256]),('conv5_4_bn/moving_variance', [256]),
    ('conv5_5/kernel', [3, 3, 256, 512]),('conv5_5_bn/gamma', [512]),('conv5_5_bn/beta', [512]),('conv5_5_bn/moving_mean', [512]),('conv5_5_bn/moving_variance',[512]),
    ('conv6_1/kernel', [3, 3, 512, 1024]),('conv6_1_bn/gamma', [1024]),('conv6_1_bn/beta', [1024]),('conv6_1_bn/moving_mean', [1024]),('conv6_1_bn/moving_variance', [1024]),
    ('conv6_2/kernel', [1, 1, 1024, 512]),('conv6_2_bn/gamma', [512]),('conv6_2_bn/beta', [512]),('conv6_2_bn/moving_mean', [512]),('conv6_2_bn/moving_variance', [512]),
    ('conv6_3/kernel', [3, 3, 512, 1024]),('conv6_3_bn/gamma', [1024]),('conv6_3_bn/beta', [1024]),('conv6_3_bn/moving_mean', [1024]),('conv6_3_bn/moving_variance', [1024]),
    ('conv6_4/kernel', [1, 1, 1024, 512]),('conv6_4_bn/gamma', [512]),('conv6_4_bn/beta', [512]),('conv6_4_bn/moving_mean', [512]),('conv6_4_bn/moving_variance', [512]),
    ('conv6_5/kernel', [3, 3, 512, 1024]),('conv6_5_bn/gamma', [1024]),('conv6_5_bn/beta', [1024]),('conv6_5_bn/moving_mean', [1024]),('conv6_5_bn/moving_variance', [1024]),
    ('conv7_1/kernel', [3, 3, 1024,1024]),('conv7_1_bn/gamma', [1024]),('conv7_1_bn/beta', [1024]),('conv7_1_bn/moving_mean', [1024]), ('conv7_1_bn/moving_variance', [1024]),
    ('conv7_2/kernel', [3, 3, 1024, 1024]),('conv7_2_bn/gamma', [1024]),('conv7_2_bn/beta', [1024]),('conv7_2_bn/moving_mean', [1024]),('conv7_2_bn/moving_variance', [1024]),
    ('conv_shortcut/kernel', [1, 1, 512, 64]),('conv_shortcut_bn/gamma', [64]),('conv_shortcut_bn/beta', [64]),('conv_shortcut_bn/moving_mean',[64]),('conv_shortcut_bn/moving_variance', [64]),
    ('conv8/kernel', [3, 3, 1280, 1024]),('conv8_bn/gamma', [1024]), ('conv8_bn/beta', [1024]),('conv8_bn/moving_mean', [1024]),('conv8_bn/moving_variance', [1024]),
    ('conv_dec/kernel', [1, 1, 1024, 425]),('conv_dec/bias', [425])
]

modelWeightsNames=[
    ('conv1/weights:0',[3, 3, 3, 32]),('conv1/BatchNorm/gamma:0',[32]),('conv1/BatchNorm/beta:0',[32]),('conv1/BatchNorm/moving_mean:0',[32]),('conv1/BatchNorm/moving_variance:0',[32]),
    ('conv2/weights:0',[3, 3, 32, 64]),('conv2/BatchNorm/gamma:0',[64]),('conv2/BatchNorm/beta:0',[64]),('conv2/BatchNorm/moving_mean:0',[64]),('conv2/BatchNorm/moving_variance:0',[64]),
    ('conv3_1/weights:0',[3, 3, 64, 128]),('conv3_1/BatchNorm/gamma:0',[128]),('conv3_1/BatchNorm/beta:0',[128]),('conv3_1/BatchNorm/moving_mean:0', [128]),('conv3_1/BatchNorm/moving_variance:0',[128]),
    ('conv3_2/weights:0',[1, 1, 128, 64]),('conv3_2/BatchNorm/gamma:0',[64]),('conv3_2/BatchNorm/beta:0',[64]),('conv3_2/BatchNorm/moving_mean:0',[64]),('conv3_2/BatchNorm/moving_variance:0',[64]),
    ('conv3_3/weights:0',[3, 3, 64, 128]),('conv3_3/BatchNorm/gamma:0',[128]),('conv3_3/BatchNorm/beta:0',[128]),('conv3_3/BatchNorm/moving_mean:0',[128]),('conv3_3/BatchNorm/moving_variance:0',[128]),
    ('conv4_1/weights:0',[3, 3, 128, 256]),('conv4_1/BatchNorm/gamma:0',[256]),('conv4_1/BatchNorm/beta:0',[256]),('conv4_1/BatchNorm/moving_mean:0',[256]),('conv4_1/BatchNorm/moving_variance:0',[256]),
    ('conv4_2/weights:0',[1, 1, 256, 128]),('conv4_2/BatchNorm/gamma:0',[128]), ('conv4_2/BatchNorm/beta:0',[128]),('conv4_2/BatchNorm/moving_mean:0',[128]),('conv4_2/BatchNorm/moving_variance:0',[128]),
    ('conv4_3/weights:0',[3, 3, 128, 256]),('conv4_3/BatchNorm/gamma:0',[256]),('conv4_3/BatchNorm/beta:0',[256]),('conv4_3/BatchNorm/moving_mean:0',[256]),('conv4_3/BatchNorm/moving_variance:0',[256]),
    ('conv5_1/weights:0',[3, 3, 256, 512]),('conv5_1/BatchNorm/gamma:0',[512]),('conv5_1/BatchNorm/beta:0',[512]),('conv5_1/BatchNorm/moving_mean:0',[512]),('conv5_1/BatchNorm/moving_variance:0',[512]),
    ('conv5_2/weights:0',[1, 1, 512, 256]),('conv5_2/BatchNorm/gamma:0',[256]),('conv5_2/BatchNorm/beta:0',[256]),('conv5_2/BatchNorm/moving_mean:0',[256]),('conv5_2/BatchNorm/moving_variance:0',[256]),
    ('conv5_3/weights:0',[3, 3, 256, 512]),('conv5_3/BatchNorm/gamma:0',[512]),('conv5_3/BatchNorm/beta:0',[512]),('conv5_3/BatchNorm/moving_mean:0',[512]),('conv5_3/BatchNorm/moving_variance:0',[512]),
    ('conv5_4/weights:0',[1, 1, 512, 256]),('conv5_4/BatchNorm/gamma:0',[256]),('conv5_4/BatchNorm/beta:0',[256]),('conv5_4/BatchNorm/moving_mean:0',[256]),('conv5_4/BatchNorm/moving_variance:0',[256]),
    ('conv5_5/weights:0',[3, 3, 256, 512]),('conv5_5/BatchNorm/gamma:0',[512]),('conv5_5/BatchNorm/beta:0',[512]),('conv5_5/BatchNorm/moving_mean:0',[512]),('conv5_5/BatchNorm/moving_variance:0',[512]),
    ('conv_shortcut/weights:0',[1, 1, 512, 64]),('conv_shortcut/BatchNorm/gamma:0',[64]),('conv_shortcut/BatchNorm/beta:0',[64]),('conv_shortcut/BatchNorm/moving_mean:0',[64]),('conv_shortcut/BatchNorm/moving_variance:0',[64]),
    ('conv6_1/weights:0',[3, 3, 512, 1024]),('conv6_1/BatchNorm/gamma:0',[1024]),('conv6_1/BatchNorm/beta:0',[1024]),('conv6_1/BatchNorm/moving_mean:0',[1024]), ('conv6_1/BatchNorm/moving_variance:0',[1024]),
    ('conv6_2/weights:0',[1, 1, 1024, 512]),('conv6_2/BatchNorm/gamma:0',[512]),('conv6_2/BatchNorm/beta:0',[512]),('conv6_2/BatchNorm/moving_mean:0',[512]),('conv6_2/BatchNorm/moving_variance:0',[512]),
    ('conv6_3/weights:0',[3, 3, 512, 1024]),('conv6_3/BatchNorm/gamma:0',[1024]),('conv6_3/BatchNorm/beta:0',[1024]),('conv6_3/BatchNorm/moving_mean:0',[1024]),('conv6_3/BatchNorm/moving_variance:0',[1024]),
    ('conv6_4/weights:0',[1, 1, 1024, 512]),('conv6_4/BatchNorm/gamma:0',[512]),('conv6_4/BatchNorm/beta:0',[512]),('conv6_4/BatchNorm/moving_mean:0',[512]),('conv6_4/BatchNorm/moving_variance:0',[512]),
    ('conv6_5/weights:0',[3, 3, 512, 1024]),('conv6_5/BatchNorm/gamma:0',[1024]),('conv6_5/BatchNorm/beta:0',[1024]),('conv6_5/BatchNorm/moving_mean:0',[1024]),('conv6_5/BatchNorm/moving_variance:0',[1024]),
    ('conv7_1/weights:0',[3, 3, 1024, 1024]),('conv7_1/BatchNorm/gamma:0',[1024]),('conv7_1/BatchNorm/beta:0',[1024]),('conv7_1/BatchNorm/moving_mean:0',[1024]),('conv7_1/BatchNorm/moving_variance:0',[1024]),
    ('conv7_2/weights:0',[3, 3, 1024, 1024]),('conv7_2/BatchNorm/gamma:0',[1024]),('conv7_2/BatchNorm/beta:0',[1024]),('conv7_2/BatchNorm/moving_mean:0',[1024]),('conv7_2/BatchNorm/moving_variance:0',[1024]),
    ('conv8/weights:0',[3, 3, 1280, 1024]),('conv8/BatchNorm/gamma:0',[1024]),('conv8/BatchNorm/beta:0',[1024]),('conv8/BatchNorm/moving_mean:0',[1024]),('conv8/BatchNorm/moving_variance:0',[1024]),
    ('conv_dec/weights:0',[1, 1, 1024, 425]),('conv_dec/biases:0',[425])
]

anchors = [[0.57273, 0.677385],
            [1.87446, 2.06253],
           [3.33843, 5.47434],
           [7.88282, 3.52778],
           [9.77052, 9.16828]]

classNames=['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
            'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
            'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
            'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
            'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
            'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
            'pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard',
            'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
            'teddy bear','hair drier','toothbrush']

multiScaleInputsize = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]   # w, h
multiScaleOutsize = [multiScaleInputsize[0] / 32,
                        multiScaleInputsize[1] / 32,
                        multiScaleInputsize[2] / 32,
                        multiScaleInputsize[3] / 32,
                        multiScaleInputsize[4] / 32,
                        multiScaleInputsize[5] / 32,
                        multiScaleInputsize[6] / 32,
                        multiScaleInputsize[7] / 32,
                        multiScaleInputsize[8] / 32,
                        # multi_scale_inp_size[9] / 32,
                        ]   # w, h
inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32
