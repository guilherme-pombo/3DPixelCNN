import os
import numpy as np

from keras.layers import Input, multiply, add
from keras.layers.core import Lambda, SpatialDropout3D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers.convolutional import ZeroPadding3D
from keras.layers.convolutional import Convolution3D
from keras import backend as K


class GatedCNN3D(object):
    ''' Convolution layer with gated activation unit. '''

    def __init__(self, nb_filters, stack_name, v_map=None, d_map=None, crop_right=False):
        '''
        Args:
            nb_filters (int)         : Number of the filters (feature maps)
            stack_name (str)		: 'vertical' or 'horizontal'
            v_map (numpy.ndarray)   : Vertical maps if feeding into horizontal stack. (default:None)
            crop_right (bool)       : if True, crop rightmost of the feature maps (mask A, introduced in [https://arxiv.org/abs/1601.06759] )
        '''
        self.nb_filters = nb_filters
        self.stack_name = stack_name
        self.v_map = v_map
        self.d_map = d_map
        self.crop_right = crop_right

    @staticmethod
    def _crop_right(x):
        return x[:, :, :, :-1, :]

    @staticmethod
    def _crop_depth(x):
        x_shape = K.int_shape(x)
        return x[:, :, :, :x_shape[3] - 1, :]

    def __call__(self, xW, layer_idx):
        '''calculate gated activation maps given input maps '''
        if self.stack_name == 'vertical':
            stack_tag = 'v'
        elif self.stack_name == 'horizontal':
            stack_tag = 'h'
        else:
            stack_tag = 'd'

        # CROP SO THAT FEED MAPS HAVE THE SAME SHAPE
        if self.crop_right:
            xW = Lambda(self._crop_right, name='h_crop_right_' + str(layer_idx))(xW)

        # ADD THE FEEDMAPS - horizontal looks at vertical and depth
        if self.v_map is not None and self.d_map is not None:
            xW = add([xW, self.v_map, self.d_map], name='h_merge_v_' + str(layer_idx))

        # vertical only looks at depth
        if self.d_map is not None and self.v_map is None:
            xW = add([xW, self.d_map], name='v_merge_d_' + str(layer_idx))

        # get first N filters -- apply tanh after
        xW_f = Lambda(lambda x: x[:, :, :, :, :self.nb_filters], name=stack_tag + '_Wf_' + str(layer_idx))(xW)
        # get everything after those filters -- apply sigmoid after
        xW_g = Lambda(lambda x: x[:, :, :, :, self.nb_filters:], name=stack_tag + '_Wg_' + str(layer_idx))(xW)

        xW_f = Lambda(lambda x: K.tanh(x), name=stack_tag + '_tanh_' + str(layer_idx))(xW_f)
        xW_g = Lambda(lambda x: K.sigmoid(x), name=stack_tag + '_sigmoid_' + str(layer_idx))(xW_g)

        res = multiply([xW_f, xW_g], name=stack_tag + '_merge_gate_' + str(layer_idx))

        return res


class PixelCNN3D(object):

    def __init__(self, input_size, nb_channels=1, nb_pixelcnn_layers=13, nb_filters=128, filter_size_1st=(7, 7, 7),
                 filter_size=(3, 3, 3), optimizer='adam', es_patience=100, save_root='/tmp/pixelcnn',
                 dropout_rate=0.2, training_dropout=True):
        '''
        Args:
            input_size ((int,int))      : (height, width) pixels of input images
            nb_channels (int)           : Number of channels for input images. (1 for grayscale images, 3 for color images)
            nb_pixelcnn_layers (int)    : Number of layers (except last two ReLu layers). (default:13)
            nb_filters (int)            : Number of filters (feature maps) for each layer. (default:128)
            filter_size_1st ((int, int)): Kernel size for the first layer. (default: (7,7))
            filter_size ((int, int))    : Kernel size for the subsequent layers. (default: (3,3))
            optimizer (str)             : SGD optimizer (default: 'adadelta')
            es_patience (int)           : Number of epochs with no improvement after which training will be stopped (EarlyStopping)
            save_root (str)             : Root directory to which {trained model file, parameter.txt, tensorboard log file} are saved
            save_best_only (bool)       : if True, the latest best model will not be overwritten (default: False)
        '''
        K.set_image_dim_ordering('tf')

        self.input_size = input_size
        self.nb_pixelcnn_layers = nb_pixelcnn_layers
        self.nb_filters = nb_filters
        self.filter_size_1st = filter_size_1st
        self.filter_size = filter_size
        self.nb_channels = nb_channels
        if self.nb_channels == 1:
            self.loss = 'binary_crossentropy'
        elif self.nb_channels == 3:
            self.loss = 'categorical_crossentropy'
        self.optimizer = optimizer
        self.es_patience = es_patience

        tensorboard_dir = os.path.join(save_root, 'pixelcnn-tensorboard')
        self.tensorboard = TensorBoard(log_dir=tensorboard_dir)
        model_dir = os.path.join(save_root, '3d_pixelcnn.h5')
        self.checkpointer = ModelCheckpoint(filepath=model_dir, verbose=1, save_weights_only=True, save_best_only=True)
        self.earlystopping = EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0, mode='auto')
        self.dropout_rate = dropout_rate
        self.training_dropout = training_dropout

    def _masked_conv(self, x, filter_size, stack_name, layer_idx, mask_type='B'):

        if stack_name == 'vertical':
            # e.g. (8, 8, 8) now becomes (9, 10, 10)
            res = ZeroPadding3D(padding=((filter_size[0] // 2, 0),
                                         (filter_size[1] // 2, filter_size[1] // 2),
                                         (filter_size[2] // 2, filter_size[1] // 2)))(x)
            # back to (8, 8, 8) by using rectangular kernel (2, 3, 3)
            res = Convolution3D(filters=2 * self.nb_filters,
                                kernel_size=(filter_size[0] // 2 + 1, filter_size[1],  filter_size[2]),
                                padding='valid')(res)

        elif stack_name == 'depth':
            # e.g. (8, 8, 8) now becomes (9, 10, 10)
            res = ZeroPadding3D(padding=((0, 0),
                                (filter_size[2] // 2, 0),
                                (filter_size[2] // 2, filter_size[1] // 2)))(x)
            # back to (8, 8, 8) by using kernel (1, 2, 3)
            res = Convolution3D(filters=1,
                                kernel_size=(1, filter_size[1] // 2 + 1, filter_size[2]),
                                padding='valid')(res)

        elif stack_name == 'horizontal':
            # e.g. turn (8, 8, 8) into (8, 8, 9)
            res = ZeroPadding3D(padding=((0, 0),
                                         (0, 0),
                                         (filter_size[2] // 2, 0)))(x)
            # mask type A zeros out the center weights as well
            # so (8, 8, 9) will remain (8, 8, 9) -- have to crop it later on in Gated CNN
            if mask_type == 'A':
                res = Convolution3D(filters=2 * self.nb_filters,
                                    kernel_size=(1, 1, filter_size[2] // 2),
                                    name='h_conv_' + str(layer_idx))(res)
            # don't zero out centre so (8, 8, 9) will become (8, 8, 8)
            else:
                res = Convolution3D(filters=2 * self.nb_filters,
                                    kernel_size=(1, 1, filter_size[2] // 2 + 1),
                                    name='h_conv_' + str(layer_idx))(res)

        return res

    @staticmethod
    def _shift_depth(x):
        # Adds 1 row of zero padding and then pushes everything down 1 depth
        x_shape = K.int_shape(x)
        x = ZeroPadding3D(padding=((0, 0),
                                   (0, 0),
                                   (1, 0)))(x)
        x = Lambda(lambda x: x[:, :, :, :x_shape[3], :])(x)
        return x

    def _feed_d_map(self, x, layer_idx):
        # shifting down feature maps
        x = Lambda(self._shift_depth, name='d_shift_right' + str(layer_idx))(x)
        x = Convolution3D(filters=2 * self.nb_filters, kernel_size=(1, 1, 1), padding='valid',
                          name='d_1x1_conv_' + str(layer_idx))(x)
        return x

    @staticmethod
    def _shift_vert(x):
        # Adds 1 row of zero padding and then pushes everything down 1 row
        x_shape = K.int_shape(x)
        x = ZeroPadding3D(padding=((1, 0),
                                   (0, 0),
                                   (0, 0)))(x)
        x = Lambda(lambda x: x[:, :x_shape[1], :, :, :])(x)
        return x

    def _feed_v_map(self, x, layer_idx):
        # shifting down feature maps
        print('v_shift_down' + str(layer_idx))
        x = Lambda(self._shift_vert, name='v_shift_down' + str(layer_idx))(x)
        x = Convolution3D(filters=2 * self.nb_filters, kernel_size=(1, 1, 1), padding='valid',
                          name='feed_v_1x1x1_conv_' + str(layer_idx))(x)
        return x

    def _build_layers(self, x):
        # VERTICAL -- grows as a cube, not conditioned on any other stacks
        v_masked_map = self._masked_conv(x, self.filter_size_1st, 'vertical', 0)
        v_masked_map = SpatialDropout3D(rate=self.dropout_rate)(v_masked_map, training=self.training_dropout)
        v_stack_out = GatedCNN3D(self.nb_filters, 'vertical', d_map=None)(v_masked_map, 0)
        # This is the shifted version of the vertical map, to be use in the depth and horizontal stacks
        v_feed_map = self._feed_v_map(v_masked_map, 0)

        # DEPTH - grows as a rectangle, conditioned on the vertical stack
        d_masked_map = self._masked_conv(x, self.filter_size_1st, 'depth', 0)
        d_masked_map = SpatialDropout3D(rate=self.dropout_rate)(d_masked_map, training=self.training_dropout)
        d_stack_out = GatedCNN3D(self.nb_filters, 'depth', d_map=v_feed_map)(d_masked_map, 0)
        # This is the shifted version of the depth map, to be used in the horizontal stack
        d_feed_map = self._feed_d_map(d_stack_out, 0)
        # make it have a matching number of filters
        d_stack_out = Convolution3D(self.nb_filters, 1, padding='valid', name='v_1x1x1_conv_0')(d_stack_out)
        d_stack_out = SpatialDropout3D(rate=self.dropout_rate)(d_stack_out, training=self.training_dropout)

        # HORIZONTAL
        h_masked_map = self._masked_conv(x, self.filter_size_1st, 'horizontal', 0, 'A')
        h_masked_map = SpatialDropout3D(rate=self.dropout_rate)(h_masked_map, training=self.training_dropout)
        # horizontal stack takes in depth and vertical stacks
        # because we used a Mask of type A, now we have to crop the center element as well
        h_stack_out = GatedCNN3D(self.nb_filters, 'horizontal', v_map=v_feed_map, d_map=d_feed_map,
                                 crop_right=True)(h_masked_map, 0)
        # no residual connection in the first layer.
        h_stack_out = Convolution3D(self.nb_filters, 1, padding='valid', name='h_1x1x1_conv_0')(h_stack_out)
        h_stack_out = SpatialDropout3D(rate=self.dropout_rate)(h_stack_out, training=self.training_dropout)

        # subsequent PixelCNN layers
        for i in range(1, self.nb_pixelcnn_layers):
            # VERTICAL
            v_masked_map = self._masked_conv(v_stack_out, self.filter_size, 'vertical', i)
            v_masked_map = SpatialDropout3D(rate=self.dropout_rate)(v_masked_map, training=self.training_dropout)
            v_stack_out = GatedCNN3D(self.nb_filters, 'vertical', d_map=None)(v_masked_map, i)
            v_feed_map = self._feed_v_map(v_masked_map, i)

            # DEPTH
            d_stack_out_prev = d_stack_out
            d_masked_map = self._masked_conv(d_stack_out, self.filter_size, 'depth', i)
            d_masked_map = SpatialDropout3D(rate=self.dropout_rate)(d_masked_map, training=self.training_dropout)
            d_stack_out = GatedCNN3D(self.nb_filters, 'depth', d_map=v_feed_map)(d_masked_map, i)
            d_feed_map = self._feed_d_map(d_stack_out, i)

            d_stack_out = Convolution3D(self.nb_filters, 1, padding='valid', name='v_1x1x1_conv_' + str(i))(d_stack_out)
            d_stack_out = SpatialDropout3D(rate=self.dropout_rate)(d_stack_out, training=self.training_dropout)
            # Add a residual connection to the previous depth stack
            d_stack_out = add([d_stack_out, d_stack_out_prev], name='d_residual_' + str(i))

            # HORIZONTAL
            # use this shortcut for residual connection
            h_stack_out_prev = h_stack_out
            h_masked_map = self._masked_conv(h_stack_out, self.filter_size, 'horizontal', i)
            h_masked_map = SpatialDropout3D(rate=self.dropout_rate)(h_masked_map, training=self.training_dropout)
            # Now we are using Mask B no need to crop the center pixel
            h_stack_out = GatedCNN3D(self.nb_filters, 'horizontal', v_map=v_feed_map, d_map=d_feed_map)(h_masked_map, i)
            h_stack_out = Convolution3D(self.nb_filters, 1, padding='valid', name='h_1x1x1_conv_' + str(i))(h_stack_out)
            h_stack_out = SpatialDropout3D(rate=self.dropout_rate)(h_stack_out, training=self.training_dropout)
            # Add a residual connection to the previous horizontal stack
            h_stack_out = add([h_stack_out, h_stack_out_prev], name='h_residual_' + str(i))

        # FINAL LAYERS
        h_stack_out = Convolution3D(self.nb_filters, 1, activation='relu', padding='valid', name='penultimate_convs0')(h_stack_out)
        h_stack_out = SpatialDropout3D(rate=self.dropout_rate)(h_stack_out, training=self.training_dropout)
        h_stack_out = Convolution3D(self.nb_filters, 1, activation='relu', padding='valid', name='penultimate_convs1')(h_stack_out)
        h_stack_out = SpatialDropout3D(rate=self.dropout_rate)(h_stack_out, training=self.training_dropout)

        # We're using a low number of filters at the end, so that embeddings are not massive
        h_stack_out = Convolution3D(10, 1, activation='relu', padding='valid', name='penultimate_convs2')(h_stack_out)
        h_stack_out = SpatialDropout3D(rate=self.dropout_rate)(h_stack_out, training=self.training_dropout)

        # Finally project it back into the original volume domain
        res = Convolution3D(1, 1, activation='sigmoid', padding='valid')(h_stack_out)
        return res

    def build_model(self):
        ''' build conditional PixelCNN model '''
        if self.nb_channels == 1:
            input_img = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2], 1),
                              name='grayscale_image')
            # input_img = SpatialDropout3D(rate=self.dropout_rate)(input_img, training=self.training_dropout)
        elif self.nb_channels == 3:
            input_img = Input(shape=(self.input_size[0], self.input_size[1], self.input_size[2], 3),
                              name='color_image')

        predicted = self._build_layers(input_img)
        self.model = Model(input_img, predicted)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, x, y, batch_size, nb_epoch, validation_data=None, shuffle=True):
        self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping],
            validation_data=validation_data,
            shuffle=shuffle
        )

    def fit_generator(self, train_generator, samples_per_epoch, nb_epoch, validation_data=None, nb_val_samples=10000):
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            callbacks=[self.tensorboard, self.checkpointer, self.earlystopping,
                       ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)],
            validation_data=validation_data,
            validation_steps=nb_val_samples
        )

    def save_val_preds(self, val_batch, preds_fp, true_fp):
        true_vals = val_batch[0]
        preds = self.model.predict(true_vals)
        np.save(file=preds_fp, arr=preds)
        np.save(file=true_fp, arr=true_vals)

    def load_model(self, checkpoint_file):
        self.model = load_model(checkpoint_file)

    @classmethod
    def predict(self, x, batch_size):
        return self.model.predict(x, batch_size)


if __name__ == '__main__':
    pass
