import numpy as np
import scipy as sp
from scipy import misc, interpolate
import time

from chainer import cuda, optimizers, Variable
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe

gpu_flag = False
xp = np


class VGG19:
    def __init__(self):
        print("Loading Model...")
        start_time = time.time()
        # self.vgg = L.caffe.CaffeFunction('VGG_ILSVRC_19_layers.caffemodel')
        self.vgg = L.caffe.CaffeFunction('vgg_normalised.caffemodel')
        if gpu_flag:
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
    def __init__(self, neural_net, content_image, style_image, content_img_chr, alpha=50.0, beta=10000.0,
                 keep_color=False):
        self.neural_net = neural_net

        self.preserve_color = keep_color  # flag for preserving color

        self.alpha = alpha  # weighting factors for content
        self.beta = beta  # weighting factors for style

        self.content_img = Variable(xp.zeros_like(content_image.data))
        self.style_img = Variable(xp.zeros_like(style_image.data))
        self.content_img_chr = Variable(xp.zeros_like(content_image.data))

        self.content_img.copydata(content_image)
        self.style_img.copydata(style_image)

        if keep_color:
            self.content_img_chr.copydata(content_img_chr)

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
                      iterations=2000, save=50, filename='iter', str_contrast=False):
        chainer_adam = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=eps)
        chainer_adam.t = 0
        state = {'m': xp.zeros_like(init_img.data), 'v': xp.zeros_like(init_img.data)}
        out_img = Variable(xp.zeros_like(init_img.data), volatile=True)

        time_start = time.time()
        for epoch in range(iterations):
            chainer_adam.t += 1

            loss = self.loss_total(init_img)
            loss.backward()
            loss.unchain_backward()

            # normalize gradient
            grad_l1_norm = xp.sum(xp.absolute(init_img.grad * init_img.grad))
            init_img.grad /= grad_l1_norm

            if gpu_flag:
                chainer_adam.update_one_gpu(init_img, state)
            else:
                chainer_adam.update_one_cpu(init_img, state)

            init_img.zerograd()

            # save image every 'save' iteration
            if save != 0 and (epoch + 1) % save == 0:
                if self.preserve_color:
                    init_img_lum = separate_lum_chr(init_img)[0]
                    if gpu_flag:
                        init_img_lum.to_gpu()
                    out_img.copydata(init_img_lum + self.content_img_chr)
                else:
                    out_img.copydata(init_img)
                save_image(out_img, filename + '_' + str(epoch + 1) + '.png', contrast=str_contrast)
                print("Image Saved at Iteration %.0f, Time Used: %.4f, Total Loss: %.4f" %
                      ((epoch + 1), (time.time() - time_start), loss.data))

    def optimize_rmsprop(self, init_img, lr=0.1, alpha=0.95, momentum=0.9, eps=1e-4,
                         iterations=2000, save=50, filename='iter', str_contrast=False):
        chainer_rms = optimizers.RMSpropGraves(lr=lr, alpha=alpha, momentum=momentum, eps=eps)
        state = {'n': xp.zeros_like(init_img.data), 'g': xp.zeros_like(init_img.data),
                 'delta': xp.zeros_like(init_img.data)}
        out_img = Variable(xp.zeros_like(init_img.data), volatile=True)

        time_start = time.time()
        for epoch in range(iterations):
            loss = self.loss_total(init_img)
            loss.backward()
            loss.unchain_backward()

            # normalize gradient
            grad_l1_norm = xp.sum(xp.absolute(init_img.grad * init_img.grad))
            init_img.grad /= grad_l1_norm

            if gpu_flag:
                chainer_rms.update_one_gpu(init_img, state)
            else:
                chainer_rms.update_one_cpu(init_img, state)

            init_img.zerograd()

            # save image every 'save' iteration
            if save != 0 and (epoch + 1) % save == 0:
                if self.preserve_color:
                    init_img_lum = separate_lum_chr(init_img)[0]
                    if gpu_flag:
                        init_img_lum.to_gpu()
                    out_img.copydata(init_img_lum + self.content_img_chr)
                else:
                    out_img.copydata(init_img)
                save_image(out_img, filename + '_' + str(epoch + 1) + '.png', contrast=str_contrast)
                print("Image Saved at Iteration %.0f, Time Used: %.4f, Total Loss: %.4f" %
                      ((epoch + 1), (time.time() - time_start), loss.data))


# rescale an array to [0, 255]
def normalize(data):
    n_data = (data - data.min()) / (data.max() - data.min())

    return n_data


# convert RGB to YIQ
def rgb_to_yiq(rgb_img):
    yiq_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    yiq_img = np.dot(normalize(rgb_img), yiq_mat.T).astype(np.float32)

    # separate luminance and chrominance channel
    yiq_img_lum = np.zeros_like(yiq_img)
    yiq_img_chr = np.zeros_like(yiq_img)

    yiq_img_lum[:, :, 0] = yiq_img[:, :, 0]
    yiq_img_chr[:, :, 1:3] = yiq_img[:, :, 1:3]

    return yiq_img_lum, yiq_img_chr


# convert YIQ to RGB
def yiq_to_rgb(yiq_img):
    rgb_mat = np.array([[1.000, 0.956, 0.621], [1.000, -0.273, -0.647], [1.000, -1.104, 1.701]])
    rgb_img = np.dot(yiq_img, rgb_mat.T).astype(np.float32)

    # normalize to [0, 255]
    rgb_img = 255.0 * normalize(rgb_img)

    return rgb_img


# separate luminance and chrominance channel
def separate_lum_chr(gen_img_cvar):
    gen_img = gen_img_cvar.data.copy()
    if gpu_flag:
        gen_img = xp.asnumpy(gen_img)

    # roll back to standard arrangement and flip to RGB
    gen_img = np.rollaxis(np.squeeze(gen_img, 0), 0, 3)[..., ::-1]

    # separate channel
    gen_img_lum, gen_img_chr = rgb_to_yiq(gen_img)
    gen_img_lum = yiq_to_rgb(gen_img_lum)
    gen_img_chr = yiq_to_rgb(gen_img_chr)

    # flip to BGR
    gen_img_lum = gen_img_lum[..., ::-1]
    gen_img_chr = gen_img_chr[..., ::-1]

    # convert to Chainer Variables
    gen_img_lum = Variable(gen_img_lum)
    gen_img_chr = Variable(gen_img_chr)

    # transform images into bc01 arrangement
    gen_img_lum = F.rollaxis(gen_img_lum, 2, 0)[np.newaxis, ...]
    gen_img_chr = F.rollaxis(gen_img_chr, 2, 0)[np.newaxis, ...]

    return gen_img_lum, gen_img_chr


# match two images using the Monge-Kantorovitch transform
def histogram_match(cont_img_cvar, sty_img_cvar):
    cont_img = cont_img_cvar.data.copy()
    sty_img = sty_img_cvar.data.copy()

    # roll back to standard arrangement
    cont_img = np.rollaxis(np.squeeze(cont_img, 0), 0, 3)
    sty_img = np.rollaxis(np.squeeze(sty_img, 0), 0, 3)

    # compute row means
    cont_mu = np.mean(cont_img, axis=(0, 1))
    sty_mu = np.mean(sty_img, axis=(0, 1))

    # compute covariance matrix
    cont_sigma = np.cov(np.concatenate(cont_img), rowvar=False, bias=True)
    sty_sigma = np.cov(np.concatenate(sty_img), rowvar=False, bias=True)

    # eigendecomposition for square roots
    sty_q, sty_l = sp.linalg.eig(sty_sigma)
    sty_q = np.diag(np.sqrt(sty_q))
    sty_sigma_sqrtm = sty_l.dot(sty_q).dot(sty_l.T)
    sty_sigma_sqrtm_inv = np.linalg.inv(sty_sigma_sqrtm)

    cont_sty_cov = sty_sigma_sqrtm.dot(cont_sigma).dot(sty_sigma_sqrtm)
    cs_q, cs_l = sp.linalg.eig(cont_sty_cov)
    cs_q = np.diag(np.sqrt(cs_q))
    cs_sqrtm = cs_l.dot(cs_q).dot(cs_l.T)

    # color matching transformation
    a = sty_sigma_sqrtm_inv.dot(cs_sqrtm).dot(sty_sigma_sqrtm_inv)
    sty_img_col = np.add(np.dot(sty_img - sty_mu, a.T), cont_mu).real

    # normalize
    sty_img_col = np.ceil(255.0 * normalize(sty_img_col))

    # convert to a Chainer Variables
    sty_img_cm_cvar = Variable(sty_img_col)

    # transform image back to bc01
    sty_img_cm_cvar = F.rollaxis(sty_img_cm_cvar, 2, 0)[np.newaxis, ...]

    return sty_img_cm_cvar


# load content and style from files
def load_images(content_name, style_name):
    # load images as arrays
    content_img = sp.misc.imread(content_name, mode='RGB').astype(np.float32)
    style_img = sp.misc.imread(style_name, mode='RGB').astype(np.float32)
    style_img = sp.misc.imresize(style_img, size=content_img.shape[0:2], interp='lanczos').astype(np.float32)

    # flip to BGR
    content_img = content_img[..., ::-1]
    style_img = style_img[..., ::-1]

    # convert to Chainer Variables
    content_img = Variable(content_img)
    style_img = Variable(style_img)

    # transform loaded images into bc01 arrangement
    content_img = F.rollaxis(content_img, 2, 0)[np.newaxis, ...]
    style_img = F.rollaxis(style_img, 2, 0)[np.newaxis, ...]

    return content_img, style_img


# write generated image to file
# gen_rep - a Chainer Variable
# filename - a string
def save_image(gen_rep, filename, contrast=False):
    mean_pixel = np.array([103.939, 116.779, 123.680]).astype(np.float32)

    out_img = gen_rep.data.copy()
    # convert to numpy array if using GPU
    if gpu_flag:
        out_img = xp.asnumpy(out_img)

    out_img = np.rollaxis(np.squeeze(out_img, 0), 0, 3)
    out_img += mean_pixel

    # flip back to RGB
    out_img = out_img[..., ::-1]

    # contrast stretching
    if contrast:
        imin, imax = np.percentile(out_img, (1, 99))
        out_img = np.clip(out_img, imin, imax)

    # normalize to [0, 255]
    out_img = 255.0 * normalize(out_img)

    sp.misc.imsave(filename, out_img, 'png')


def white_noise(orig_img):
    gen_img = xp.random.normal(size=orig_img.shape).astype(np.float32)
    gen_img = 255.0 * normalize(gen_img) - 114.80

    return gen_img


# for original VGG models mean_pixel should be subtracted
def mean_subtraction(img_cvar):
    mean_pixel = np.array([103.939, 116.779, 123.680]).astype(np.float32)

    # roll back to standard arrangement
    temp_img = np.rollaxis(np.squeeze(img_cvar.data.copy(), 0), 0, 3)

    if gpu_flag:
        temp_img = xp.asnumpy(temp_img)

    temp_img -= mean_pixel

    temp_cvar = Variable(temp_img)
    temp_cvar = F.rollaxis(temp_cvar, 2, 0)[np.newaxis, ...]

    return temp_cvar


# helper function for changing global GPU flag
def use_gpu(gpu=True):
    global gpu_flag
    global xp
    if gpu:
        gpu_flag = True
        cuda.get_device().use()
        xp = cuda.cupy
    else:
        gpu_flag = False
        xp = np


# helper function for synthesizing image
def generate_image(cnn, content, style, alpha=150.0, beta=10000.0, color='none', lum_match=True, init_image='noise',
                   optimizer='adam', iteration=1500, lr=0.15, save=50, prefix='temp', contrast=True):
    # load images
    content_img, style_img = load_images(content, style)
    content_img_chr = Variable(xp.zeros_like(content_img.data))

    # choose color preserving scheme
    color_flag = False
    if color != 'none':
        if color == 'histogram':
            style_img.copydata(histogram_match(content_img, style_img))
        elif color == 'luminance':
            color_flag = True
            content_img_lum, content_img_chr = separate_lum_chr(content_img)
            content_img_chr = mean_subtraction(content_img_chr)
            if lum_match:
                style_img_lum, style_img_chr = separate_lum_chr(style_img)
                style_img_lum.copydata(histogram_match(content_img_lum, style_img_lum))
                style_img_temp = style_img_lum + style_img_chr
                style_img_temp.data = 255.0 * normalize(style_img_temp.data)
                style_img.copydata(style_img_temp)
        else:
            return

    # subtract means before passing
    content_img = mean_subtraction(content_img)
    style_img = mean_subtraction(style_img)

    if gpu_flag:
        content_img.to_gpu()
        style_img.to_gpu()
        content_img_chr.to_gpu()

    # instantiation
    print("\nInitializing...")
    start_time_1 = time.time()
    art_nn = ArtNN(cnn, content_img, style_img, content_img_chr=content_img_chr, alpha=alpha, beta=beta,
                   keep_color=color_flag)
    print("Done. Time Used: %.2f" % (time.time() - start_time_1))

    # choose initializing image
    if init_image == 'noise':
        x = white_noise(content_img)
    elif init_image == 'content':
        x = content_img.data.copy()
    elif init_image == 'style':
        x = style_img.data.copy()
    else:
        return

    x = Variable(x)
    if gpu_flag:
        x.to_gpu()

    # generate image
    print("\nContent Image: " + content)
    print("  Style Image: " + style)

    print("\nSynthesizing...")
    print("Initial Loss: %.4f" % art_nn.loss_total(x).data)
    start_time_1 = time.time()

    # choose optimizer
    if optimizer == 'adam':
        art_nn.optimize_adam(x, iterations=iteration, alpha=lr, save=save, filename=prefix, str_contrast=contrast)
    elif optimizer == 'rmsprop':
        art_nn.optimize_rmsprop(x, iterations=iteration, lr=lr, save=save, filename=prefix, str_contrast=contrast)
    else:
        return

    print("Done. Total Time Used: %.2f" % (time.time() - start_time_1))
    print("End Loss: %.4f" % art_nn.loss_total(x).data)


def main():
    use_gpu(True)
    cnn = VGG19()

    generate_image(cnn, 'content.jpg', 'style.jpg', alpha=150.0, beta=12000.0,
                   init_image='noise', optimizer='rmsprop', iteration=1600, lr=0.25, prefix='temp')

if __name__ == "__main__":
    main()
