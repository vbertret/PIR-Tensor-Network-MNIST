# PIR Tensor Network MNIST

The aim of this study is to reproduce the experience from Miles Stoudenmire and David Schwab described in the publication "Supervised Learning with Tensor Networks" (http://arxiv.org/abs/1605.05775). 

The project aims to study the classification of images from the MNIST database which contains handwritten digits using a tensor network. For that, it uses a tensor decomposition training with basic Machine Learning algorithms, such as gradient descent, but also more sophisticated physics-inspired algorithms. 

# Description of the project

# Organisation
* `\data`-- the data which is processed or external
  * `\external` MNIST Database without any traitment
  * `\processed` modified MNIST Database with pictures of 14 pixels by 14 pixels
* `\docs` all the documentation used for the projects
* `\notebooks` different notebooks with example of code to illustrate the projects and to have some example of the syntax in order to use the package
* `\reports` the 2 reports made for the projects (the first report is in French)
  * `\figures` illustrations of reports
* `\src` the package developed to train a tensor network. Each method or class is documented
  * `\data` some method to create random data or to load the MNIST dataset
  * `\mapping` methods to compute the mapping function 
  * `\tensor` one method to contract a tensor with inputs
  * `\tensor_network` all the stuff needeed to train a tensor network
    * `modelMPS.py` the central class of the project. It defines a tensor network with different methods to train it
    * `optimizer.py` different algorithm to reduce the loss : Adam, Gradient Descent with fixed stepsize and Conjugate Gradient Descente
    * `\algo_AGD` some methods to compute the gradient and the cost for basic Alternating Gradient Descent
    * `\algos_DMRG` some methods to compute the gradient and the cost with Density Matrix Renormalization Group






