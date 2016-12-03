import numpy as np
import scipy as sp
from scipy import misc
import time

from chainer import cuda, optimizers, Variable
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe

use_gpu = False
xp = np


class VGG19:
    def __init__(self):
        print("Loading Model...")
        start_time = time.time()
        self.vgg = L.caffe.CaffeFunction('vgg_normalised.caffemodel')
        if use_gpu:
            self.vgg.to_gpu()
        print("Done. Time Used: %.2f" % (time.time() - start_time))

    def __call__(self, x):
        conv1_1 = F.relu(self.vgg.conv1_1(x))
        conv1_2 = F.relu(self.vgg.conv1_2(conv1_1))
        pool1 = F.average_pooling_2d(conv1_2, 2, stride=2)

        conv2_1 = F.relu(self.vgg.conv2_1(pool1))
        conv2_2 = F.relu(self.vgg.conv2_2(conv2_1))
        pool2 = F.average_pooling_2d(conv2_2, 2, stride=2)

        conv3_1 = F.relu(self.vgg.conv3_1(pool2))
        conv3_2 = F.relu(self.vgg.conv3_2(conv3_1))
        conv3_3 = F.relu(self.vgg.conv3_3(conv3_2))
        conv3_4 = F.relu(self.vgg.conv3_4(conv3_3))
        pool3 = F.average_pooling_2d(conv3_4, 2, stride=2)

        conv4_1 = F.relu(self.vgg.conv4_1(pool3))
        conv4_2 = F.relu(self.vgg.conv4_2(conv4_1))
        conv4_3 = F.relu(self.vgg.conv4_3(conv4_2))
        conv4_4 = F.relu(self.vgg.conv4_4(conv4_3))
        pool4 = F.average_pooling_2d(conv4_4, 2, stride=2)

        conv5_1 = F.relu(self.vgg.conv5_1(pool4))

        return tuple([conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2])


class ArtNN:
    def __init__(self, neural_net, content_image, style_image, alpha=50.0, beta=10000.0,
                 presv_color=False, lum_match=True):
        self.neural_net = neural_net

        self.preserve_color = presv_color  # flag for preserving color

        self.alpha = alpha  # weighting factors for content
        self.beta = beta  # weighting factors for style

        # if choose to preserve color, extract the luminance and chrominance
        if presv_color:
            self.content_img, self.content_img_chr = separate_lum_chr(content_image)
            self.style_img = separate_lum_chr(style_image)[0]
            if lum_match:
                # luminance match
                std_c = self.content_img.data.std()
                mu_c = self.content_img.data.mean()
                std_s = self.style_img.data.std()
                mu_s = self.style_img.data.std()

                self.style_img.data = (std_c / std_s) * (self.style_img.data - mu_s) + mu_c
        else:
            self.content_img = content_image
            self.style_img = style_image
            self.content_img_chr = 0

        self.content_rep = self.neural_net(self.content_img)[-1:]
        self.style_rep = self.neural_net(self.style_img)[:-1]

        self.content_feat_map = self.feature_map(self.content_rep)
        self.style_feat_cor = self.feature_cor(self.style_rep)

    # extract feature map from a filtered image
    @staticmethod
    def feature_map(filtered_reps):
        feat_map_list = []
        for rep in filtered_reps:
            num_channel = rep.shape[1]
            feat_map = F.reshape(rep, (num_channel, -1))
            feat_map_list.append(feat_map)

        return tuple(feat_map_list)

    # compute feature correlations of a filtered image,
    # correlations are given by the Gram matrix
    # cf. equation (3) of the article
    def feature_cor(self, filtered_reps):
        gram_mat_list = []
        feat_map_list = self.feature_map(filtered_reps)
        for feat_map in feat_map_list:
            gram_mat = F.matmul(feat_map, feat_map, transa=False, transb=True)
            gram_mat_list.append(gram_mat)

        return tuple(gram_mat_list)

    # content loss function
    # cf. equation (1) of the article
    def loss_content(self, gen_img_rep):
        feat_map_gen = self.feature_map(gen_img_rep)
        feat_loss = F.mean_squared_error(self.content_feat_map[0], feat_map_gen[0]) / 2.0

        return feat_loss

    # style loss function
    # cf. equation (5) of the article
    def loss_style(self, gen_img_rep):
        feat_cor_gen = self.feature_cor(gen_img_rep)

        feat_loss = 0
        for i in range(len(feat_cor_gen)):
            orig_shape = self.style_rep[i].shape
            feat_map_size = orig_shape[2] * orig_shape[3]  # M_l

            layer_wt = 4.0 * feat_map_size ** 2.0

            feat_loss += F.mean_squared_error(self.style_feat_cor[i], feat_cor_gen[i]) / layer_wt

        return feat_loss

    # total loss function
    # cf. equation (7) of the article
    def loss_total(self, input_img):
        input_img_rep = self.neural_net(input_img)

        content_loss = self.loss_content(input_img_rep[-1:])
        style_loss = self.loss_style(input_img_rep[:-1])

        total_loss = self.alpha * content_loss + self.beta * style_loss

        return total_loss

    def optimize_adam(self, init_img, alpha=0.5, beta1=0.9, beta2=0.999, eps=1e-8,
                      norm_grad=True, iterations=2000, save=50, filename='iter', str_contrast=True):
        chainer_adam = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=eps)
        chainer_adam.t = 0
        state = {'m': xp.zeros_like(init_img.data), 'v': xp.zeros_like(init_img.data)}

        time_start = time.time()
        for epoch in range(iterations):
            chainer_adam.t += 1

            loss = self.loss_total(init_img)
            loss.backward()
            loss.unchain_backward()

            # normalize gradient
            if norm_grad:
                grad_l1_norm = xp.sum(xp.absolute(init_img.grad * init_img.grad))
                init_img.grad /= grad_l1_norm

            if use_gpu:
                chainer_adam.update_one_gpu(init_img, state)
            else:
                chainer_adam.update_one_cpu(init_img, state)

            init_img.zerograd()

            if save != 0 and (epoch + 1) % save == 0:
                if self.preserve_color:
                    out_img = Variable(init_img.data + self.content_img_chr.data)
                else:
                    out_img = Variable(init_img.data)
                save_img(out_img, filename + '_' + str(epoch + 1) + '.png', contrast=str_contrast)
                print("Image Saved at Iteration %.0f, Time Used: %.4f, Total Loss: %.4f" %
                      ((epoch + 1), (time.time() - time_start), loss.data))

    def optimize_rmsprop(self, init_img, lr=0.1, alpha=0.95, momentum=0.9, eps=1e-4,
                         norm_grad=True, iterations=2000, save=50, filename='iter', str_contrast=True):
        chainer_rms = optimizers.RMSpropGraves(lr=lr, alpha=alpha, momentum=momentum, eps=eps)
        state = {'n': xp.zeros_like(init_img.data), 'g': xp.zeros_like(init_img.data),
                 'delta': xp.zeros_like(init_img.data)}

        time_start = time.time()
        for epoch in range(iterations):
            loss = self.loss_total(init_img)
            loss.backward()
            loss.unchain_backward()

            # normalize gradient
            if norm_grad:
                grad_l1_norm = xp.sum(xp.absolute(init_img.grad * init_img.grad))
                init_img.grad /= grad_l1_norm

            if use_gpu:
                chainer_rms.update_one_gpu(init_img, state)
            else:
                chainer_rms.update_one_cpu(init_img, state)

            init_img.zerograd()

            if save != 0 and (epoch + 1) % save == 0:
                if self.preserve_color:
                    out_img = Variable(init_img.data + self.content_img_chr.data)
                else:
                    out_img = Variable(init_img.data)
                save_img(out_img, filename + '_' + str(epoch + 1) + '.png', contrast=str_contrast)
                print("Image Saved at Iteration %.0f, Time Used: %.4f, Total Loss: %.4f" %
                      ((epoch + 1), (time.time() - time_start), loss.data))


# convert RGB to YIQ
def rgb_to_yiq(rgb_img):
    yiq_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    yiq_img = np.copy(rgb_img)

    for i in range(yiq_img.shape[0]):
        for j in range(yiq_img.shape[1]):
            yiq_img[i, j] = np.dot(yiq_mat, yiq_img[i, j])

    yiq_img /= 255.0

    # separate luminance and chrominance channel
    yiq_img_lum = np.zeros_like(yiq_img)
    yiq_img_chr = np.zeros_like(yiq_img)

    yiq_img_lum[:, :, 0] = yiq_img[:, :, 0]
    yiq_img_chr[:, :, 1:3] = yiq_img[:, :, 1:3]

    return yiq_img_lum, yiq_img_chr


# convert YIQ to RGB
def yiq_to_rgb(yiq_img):
    rgb_mat = np.array([[1.000, 0.956, 0.621], [1.000, -0.272, -0.647], [1.000, -1.106, 1.703]])
    rgb_img = np.copy(yiq_img)

    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            rgb_img[i, j] = np.dot(rgb_mat, rgb_img[i, j])

    rgb_img *= 255.0

    return rgb_img


# separate luminance and chrominance channel
def separate_lum_chr(gen_img):
    gen_img = gen_img.data
    if use_gpu:
        gen_img = xp.asnumpy(gen_img)

    gen_img = np.rollaxis(np.squeeze(gen_img, 0), 0, 3)

    # separate channel
    gen_img_lum, gen_img_chr = rgb_to_yiq(gen_img)
    gen_img_lum = yiq_to_rgb(gen_img_lum)
    gen_img_chr = yiq_to_rgb(gen_img_chr)

    # convert to Chainer Variables
    if use_gpu:
        gen_img_lum = Variable(cuda.to_gpu(gen_img_lum))
        gen_img_chr = Variable(cuda.to_gpu(gen_img_chr))
    else:
        gen_img_lum = Variable(gen_img_lum)
        gen_img_chr = Variable(gen_img_chr)

    # transform images into bc01 arrangement
    gen_img_lum = F.rollaxis(gen_img_lum, 2, 0)[xp.newaxis, ...]
    gen_img_chr = F.rollaxis(gen_img_chr, 2, 0)[xp.newaxis, ...]

    return gen_img_lum, gen_img_chr


# load content and style from files
def load_image(content_name, style_name):
    # for original VGG mean_pixel should be subtracted
    mean_pixel = np.array([123.68, 116.779, 103.939]).astype(np.float32)

    # load images as arrays
    content_img = sp.misc.imread(content_name).astype(np.float32) - mean_pixel
    style_img = sp.misc.imread(style_name, mode='RGB')
    style_img = sp.misc.imresize(style_img, size=content_img.shape[0:2], interp='lanczos').astype(np.float32)
    style_img -= mean_pixel

    # convert to Chainer Variable
    if use_gpu:
        content_img = Variable(cuda.to_gpu(content_img))
        style_img = Variable(cuda.to_gpu(style_img))
    else:
        content_img = Variable(content_img)
        style_img = Variable(style_img)

    # transform loaded images into bc01 arrangement
    content_img = F.rollaxis(content_img, 2, 0)[np.newaxis, ...]
    style_img = F.rollaxis(style_img, 2, 0)[np.newaxis, ...]

    return content_img, style_img


# write generated image to file
# gen_rep - a Chainer Variable
# filename - a string
def save_img(gen_rep, filename, contrast=True):
    mean_pixel = np.array([123.68, 116.779, 103.939]).astype(np.float32)

    out_img = gen_rep.data
    # convert to numpy array if using GPU
    if use_gpu:
        out_img = xp.asnumpy(out_img)

    out_img = np.rollaxis(np.squeeze(out_img, 0), 0, 3) + mean_pixel

    # contrast stretching
    if contrast:
        imin, imax = np.percentile(out_img, (2, 98))
        out_img = np.clip(out_img, imin, imax)
        out_img = (out_img - imin) / float(imax - imin)

    sp.misc.imsave(filename, out_img)


def white_noise(orig_img):
    gen_img = xp.random.normal(scale=50, size=orig_img.shape).astype(np.float32)

    return gen_img


# helper function for synthesizing image
def generate_image(content, style, gpu=True, alpha=80.0, beta=1000.0, color=False, luminance=True, init_image='noise',
                   optimizer='adam', norm_grad=True, iteration=2000, lr=0.1, filename='iter', contrast=True):
    # change global flag
    global use_gpu
    global xp
    if gpu:
        use_gpu = True
        cuda.get_device().use()
        xp = cuda.cupy
    else:
        use_gpu = False
        xp = np

    # load images
    content_img, style_img = load_image(content, style)

    # instantiation
    nn = VGG19()
    print("\nInitializing...")
    start_time_1 = time.time()
    art_nn = ArtNN(nn, content_img, style_img, alpha=alpha, beta=beta, presv_color=color, lum_match=luminance)
    print("Done. Time Used: %.2f" % (time.time() - start_time_1))

    # choose initializing image
    if init_image == 'noise':
        x = white_noise(content_img)
    elif init_image == 'content':
        x = content_img.data
    elif init_image == 'style':
        x = style_img.data
    else:
        return

    if use_gpu:
        x = Variable(cuda.to_gpu(x))
    else:
        x = Variable(x)

    # generate image
    print("\nGenerating Image...")
    init_loss = art_nn.loss_total(x).data
    print("Initial Loss: %.6f" % init_loss)
    start_time_1 = time.time()

    # choose optimizer
    if optimizer == 'adam':
        art_nn.optimize_adam(x, iterations=iteration, alpha=lr, norm_grad=norm_grad,
                             filename=filename, str_contrast=contrast)
    elif optimizer == 'rmsprop':
        art_nn.optimize_rmsprop(x, iterations=iteration, lr=lr, norm_grad=norm_grad,
                                filename=filename, str_contrast=contrast)
    else:
        return

    print("Done. Time Used: %.2f" % (time.time() - start_time_1))
    end_loss = art_nn.loss_total(x).data
    print("End Loss: %.6f" % end_loss)


def main():
    generate_image('grainger2.jpg', 'starry_night.jpg', alpha=250.0, beta=10000.0, init_image='noise',
                   optimizer='adam', color=False, iteration=2000, lr=0.5, filename='star1')


if __name__ == "__main__":
    main()
