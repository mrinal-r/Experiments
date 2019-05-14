# Deep Learning Project Report

**[3 points] Effort to curate/create your dataset (at least a thousand records/images)
Preprocessing (e.g. lowering resolution), cleaning, data summary
Outliers analysis
Data augmentation (if needed)**



The dataset for this project was obtained from Project Gutenberg (https://www.gutenberg.org/). This website hosts literary works of noted authors and makes them available for consumption for free. The kind of preprocessing involved varies with the goal of the project. Since my goal was to create a model to predict words, my preprocessing steps included

1. Cleaning the text to eliminate any special characters or punctuations
2. Standardize the text so that all the alphabets are lower case. This makes it easy for the tokenizer and the embedding layer to perform their task of creating word - int mappings and learning semantic word relationships respectively. Moreover, since 'Cat' and 'cat' semantically mean the same, this does not harm the model performance in any way.
3. Transform the entire text into sequences of a fixed length and fixed step. Every subsequent sequence is the previous one shifted by one word.
4. Finally, create input features and labels dataset from the above sequences. 
Please refer to README. md for additional information on data preprocessing



**[3 points] Effort to visualize input data
Example, draw plots, visualize all images in groups
Novel visualizations/plots can get you bonus points. For example, plots involving 3 or 4 columns**



The input data nd its transformation for my project can be best explained with s sample sentence and walking through the transformation to understand what the model needs. I have provided this information in the Data preprocessing section of README.md.



**[3 points] Effort correctly split data into 3 sets
Randomization**



Most of the models for text generation that i came across did not use training, validation, and test set. The current convention is to use the entire text data for training. For testing, a random sequence is picked for the training data and the model is made to predict the next N words in the sequence. 

Similarly, randomization is also not applicable to this model since we deal with sequential data.



**[3 points] Effort to design and test various neural network architectures
Why did you choose the architecture you chose?
How does the performance change if linear regression or logistic regression is used?**



I used two baseline models that were published online. The first one was similar to mine but with more number of filter units in a layer and more number of layers in general leading to slow responsiveness. The second one used convolutional layers and Max pooling, and hence was faster.

The model that I went for was moderate in size and simple in nature with just all the necessary elements such as 

1. An embedding layer to understand semantic relationships between the words
2. An LSTM layer to capture the long term relationships between the words ina sequence
3. Dense layers to provide additional supposert and transform the outputs of the previous layers to match the dimensions of the labels

I chose thsi model because in my opinion, it had all the necessary components to help it learn. It is lightweight and takes a moderate amount of time to converge and learn with an accuracy that is comparable to the other two models.



**[3 points] Effort to evaluate your results
Discussion of predictability of your model on the data you used**



I have evaluated my model by providing all the three models with the same seed text. The sequence of words predicted by each model are in line with the training accuracies seen for those models



**[3 points] Effort to benchmark your method / results
Comparison with state-of-the-art methods**



I have benchmarked my model against two word based text prediction models that were published online. For the purpose of benchmarking, I retained the structure of the original models, fed the same data to all the three models with the same batch size and number of epochs.



**[3 points] Documentation efforts (report preparation)
Documentation of all steps above**



This repository contains README.md documenting the project in detail. I have also added the original text that was used for training along with the sequences that were generated from the proprocessing step. Also included in a different folder are trained model parameters for all three models. These were saved using callbacks and checkpoint on the training accuracy. I have also included the tokenizer that containes word to int mappings for all the unique words in the text.



**[3 points] Effort to document the training time
Relationship between training time, epochs, dataset size**



I have provided graphs of training accuracy v/s number of epochs for all the three models. As for the training time, I have noted down the time taken by individual epoch to complete. Overall, the time taken per epoch remained constant for all three models.

Model 1:

Epoch 100/100

2338/2338 [==============================] - 3s 1ms/step - loss: 2.5191 - acc: 0.3353

Model 2:

Epoch 100/100

2338/2338 [==============================] - 0s 183us/step - loss: 2.1793 - acc: 0.4299

Model 3:

Epoch 100/100

2338/2338 [==============================] - 2s 680us/step - loss: 0.9978 - acc: 0.7417

Overall, since the dataset had a modest size, the training time was comparable.
Model 1 Accuracy: 

![alt text](https://github.com/mrinal-r/Experiments/blob/master/metrics-images/model1_acc.png "Model 1 Accuracy")


Model 2 Accuracy: 

![alt text](https://github.com/mrinal-r/Experiments/blob/master/metrics-images/model2_acc.png "Model 2 Accuracy")


Model 3 Accuracy: 

![alt text](https://github.com/mrinal-r/Experiments/blob/master/metrics-images/model3_acc.png "Model 3 Accuracy")



**[3 points] Effort to study learning curves
Plots of epoch vs loss on training / validation datasets**



I don't have a validation set due to requirements of my project. Hence not applicable.



**[3 points] Effort to prepare a "reproducible" Python Notebook (.ipynb) file
a Readme.txt file that outlines the steps for reproducing everything
Hosting online is encouraged (such as GitHub)**



The reproducible Python Notebook can be found at https://github.com/mrinal-r/Experiments/blob/master/Word_Based_Text_Generator.ipynb




