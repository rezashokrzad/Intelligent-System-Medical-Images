
# Intelligent-System-Medical-Images
Attempts to absorb applications of ML in Medical Images domain 
Titles:
1. Digital image
  - Sampling and quantization, Intensity transformation, Histogram matching, Linear contrast stretching
  - Spatial Filtering, Correlation at one position (x0,y0), Correlation vs. Convolution, Zero padding, cyclic padding, Filters, Smoothing filters
  - Edge detection, Prewitt gradient kernel, Sobel kernel, Gaussian kernel, Derivatives and edge detection, Derivative of Gaussian filters, Laplacian of Gaussian (LoG)
  - Blob detector, Fast Fourier Transform, Steerable filters
2. Segmentation, detection, classification introduction
  - Ground truth, Gold / Reference standard
  - Thresholding, Automatic threshold, Otsu algorithm, Labeling, Connectivity in 2D & 3D, Region growing algorithm
  - TP, TN, FP, FN in segmentation, Combination of reference and result, Accuracy, sensitivity, specificity, Jaccard index, Dice coefficient, 
  - Structuring element, Dilation, Erosion, Opening, Closing
  - Statistical texture analysis, First-order, Histogram analysis, Local Binary Pattern, Gabor filters, Feature normalization, ROC analysis, Area Under the Curve (AUC), 
  - Detection: Template matching, Matched filter, Post-processing, Feature classification
  - Detection performance: Nomenclature, Evaluation metrics, FROC
3. Machine Learning in Medical Images
  - Supervised learning, Unsupervised learning, Semi-supervised learning, Classification vs. Regression, Generative vs. Discriminative Models, Cross validation, Parametric model, Capacity of a model, 
  - Neural Networks, Non linearity, Neurons in hidden layer, Output layer, Softmax
  - Supervised learning procedure, Loss function, Cost function, One-hot representation, Optimization, Gradient descent, Backpropagation, mini-batches, Learning curves, Learning rate, sanity checks
4. Convolutional Neural Network:
  - Hubel & Wiesel, 1959 / Fukushima, 1980 / Yann LeCun, 1998 / Alex Khrizevsky, 2012
  - Convolutional Neural Network, Padding, "Valid" and "Same" convolution, Stride, Receptive Field, Fully-connected layer, Soft-max layer
  - Learned filters: AlexNet, ReLU, Initialization, Overfitting, Regularization, Dropout, Batch normalization
  - Update rules: Momentum, Nesterov Momentum, RMSProp, ADAM
  - Data augmentation, 2D Affine Transformations, Data balancing
5. VGG-net (aka OxfordNet), 2014
  - GoogLeNet, 2015 / ResNet, 2015 / DenseNet, 2016 / Using pre-trained networks
  - Effective receptive field
  - ConvNets for segmentation, Classification vs. Segmentation, Low resolution in ConvNets
  - Dilated convolutions (2016)
  - Deconvolution network, Max unpooling, Up- / De- / Transposed convolution, U-Net, 2015Bilinear interpolation
6. Detection
  - Detection of mitotic figures, Sampling strategy, Hard-negative mining, Bounding boxes, Intersection over union
  - R-CNN (November 2013), Fast R-CNN (April 2015), Faster R-CNN (June 2015), Anchor boxes
  - YOLO: You only look once (June 2015), Non‐max suppression


## the first endevour, follwoing techniques have been impleneted:
    1. conversion of raw mammography data into a gray-scale image:
            Implement the three main steps necessary to convert raw data into a gray-level image:
              - Logaritmic transformation
              - Intensity inversion
              - Contrast stretching
              
    2. stain normalization in digital pathology with histogram matching
              - In pathology, tissue samples are cut and stained with specific dyes in order to enhance some tissues that are relevant for the diagnosis.
              - The most commonly used staining is called Hematoxylyn and Eosin (H&E), which is routinely applied for diagnostic purposes.
              
    3. trachea detection in chest CT with blob detection
              
