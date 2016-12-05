## Description
An [Chainer](http://chainer.org/) implementation of *A Neural Algorithm of Artistic Style*. In short, this is an algorithm that transfers the artistic style of one image onto another by uitlizing convolutional neural networks' ability to extract high-level image content.

## Detail
### Pre-trained model
A VGG-19 Caffe model is required for this implmentation to work. You can use the normalized version used by the authors of the article: [`vgg_normalised.caffemodel`](http://bethgelab.org/deeptextures/), or use the original one made by VGG team: [`VGG_ILSVRC_19_layers.caffemodel`](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).

### Parameters
A helper function `generate_image()` was created to help the transfer. The parameters it uses are:
* `cnn`:
* `content`, `style`:
* `alpha`, `beta`:
* `color`: 
* `init_image`:
* `optimizer`: String. Optimizer to use, you can choose between `adam` for ADAM and `rmsprop` for Alex Graves’s RMSprop.
..* `iteration`: Int, number of iterations to run.
..* `lr`: Float. Learning rate of the optimizer.
* `contrast`:  


## Result
under construction ᕕ( ᐛ )ᕗ

## Reference
Leon A. Gatys, S. Ecker & Matthias Bethge (2015). [*A Neural Algorithm of Artistic Style*](http://arxiv.org/abs/1508.06576). In: *CoRR*.

Leon A. Gatys, Matthias Bethge, Aaron Hertzmann & Eli Shechtman (2016). [*Preserving Color in Neural Artistic Style Transfer*](http://arxiv.org/abs/1606.05897). In: *CoRR*.

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann & Eli Shechtman (2016). [*Controlling Perceptual Factors in Neural Style Transfer*](http://arxiv.org/abs/1611.07865). In: *CoRR*.

Gatys, Leon A., Ecker, Alexander S. & Bethge, Matthias (2016). [*Image Style Transfer Using Convolutional Neural Networks*](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html). In: *The IEEE Conference on Computer Vision and Pattern Recognition*, pp. 2414-2423.

## Acknowledgement
In making this program, I referred to helpful work of 
 
## Author
Francis Hsu, University of Illinois at Urbana–Champaign.
