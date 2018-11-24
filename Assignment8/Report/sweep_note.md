# Homework 8. Neural Network Architecture

## Jae Dong Hwang

### Highest Accuracy Achieved on *Test set* : 96.9%

I began with simple architecture of one convolution and one average sample over 500 iteraions. And after trying out several options, such as ReLu, MaxPool2d, Dropout, and Softmax, on the simple architecture, I reallearned that it's hard to make a decision on design to acheive target accuracy; 97%, without paramter sweeps. I ran several tranining and decied list of paraters and update code to run the paramter sweep.

```python
    for iteration in [500, 1000]:
        for conv1_output_channel in [6, 8, 10]:
            for conv2_output_channel in [16, 20, 24, 28]:
                for hiddenNodes in [10, 20, 40, 80]:
                    for conv_kernel_size in [3, 4, 5]:
                        for pooling_size in [2]:
                            # model training etc.
```

And I updated the model to make the model similar to LeNet to begin with and see how high the model can reach. 

* Basically, the model constructor defines the layers in following order: 

  * convolution 1 (3 x 3, 4 x 4, and 5 x 5)
  * 2 x 2 pooling 
  * convolution 2 (3 x 3, 4 x 4, and 5 x 5)
  * 2 x 2 pooling
  * full connection layer(s) (10, 20, 40, and 80 hidden nodes)

* Observation on paramter sweep results

  * Iteration 500 vs 1000
  The iteration was most outer configuration. The accuracies vary depending on configurations. Some case indicates there is still momentum to increase accuracy and others don't. Below table shows the case at 500 iteration. I ran paramter sweeps with 500 iterations and picked the case where there are still momentums to improve.

  | No Learning momenum | Learning momentum |
  |:-:|:-:|
  | ![accuracy_c1oc6_c2oc20_cksize3_psize2_hnodes10_rot_iter500](param_sweeps/accuracy_c1oc6_c2oc20_cksize3_psize2_hnodes10_rot_iter500.png)| ![accuracy_c1oc6_c2oc20_cksize4_psize2_hnodes20_rot_iter500](param_sweeps/accuracy_c1oc6_c2oc20_cksize4_psize2_hnodes20_rot_iter500.png)|

  * Activation (Sigmoid vs. ReLU)
    I've observed the sigmoid activation converage more gradually while ReLU make the convergence little bit sparsly. And since ReLU didn't help to improve accuracy, I used the sigmoid as a default.

  * Number of Down-sampling Layers
    LeNet architecture has two full connected layers. I investigated the results but found that one fully connected one with 40 hidden node produce highest results.

  * Randomness of LeNet 
    Initially, model wasn't deterministic and produced wide ranges of accuracies given a configuration. So I had to walk around by seeding the random module.

  ```python
    import torch
    # ...
    torch.manual_seed(1)
  ```

   * Data Augmentation
     I had hard time to hit the about 92.5% accuracy at most and decided the option to augment input data and use more volumn of training set. I wrote a script that reads all sample files and randomly rotate, flip, add noise to each image and save them in each associated input folder (so that I can use same label; close/open and sides).
    I increase the volumn of training set by augmenting images with three methods; rotation, noise, flipping.

  * SoftMax2D and Dropout2D
    I also tried both option on top of both convolution but didn't see the noticable changes. Dorpout2D shows the randomness as expected but didn't improve the accuracy dramatically.

* Architecutre of the Best Model
  Given the results of paramter sweeps and studying the options of activations, selections(dropout), normalization(softmax), below architecture configuration provided highest configuration on test set including randomly rotated images.

  * convolution 1 - 3 x 3 filters and 6 output channels
  * 2 x 2 max pooling 
  * convolution 2 - 3 x 3 and 16 output channels)
  * 2 x 2 max pooling
  * one full connection layer with 40 hidden nodes
  
| Accuracy | Loss |
|:-:|:-:|
|![accuracy_c1oc6_c2oc16_cksize3_psize2_hnodes40_rot_iter1000](param_sweeps/accuracy_c1oc6_c2oc16_cksize3_psize2_hnodes40_rot_iter1000.png)| ![loss_c1oc6_c2oc16_cksize3_psize2_hnodes40_rot_iter1000](param_sweeps/loss_c1oc6_c2oc16_cksize3_psize2_hnodes40_rot_iter1000.png)|

  * Statistics: 
    * Accuracy: 0.9690466364011556
    * Precision: 0.9683860232945092
    * Recall: 0.9691923397169026
    * FPR: 0.031096563011456628
    * FNR: 0.030807660283097418