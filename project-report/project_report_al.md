# Artificial Intelligence Project Report

**[2 points] Effort to study the importance of each input feature (data analysis)
For example, plot each input feature against output and document your findings**



For textual data, studying the importance of each input feature doesn't provide any insight.



**[2 points] Effort to compare the results of the neural network with a linear regression or logistic regression model?
Start with a basic model and then grow your model into a multi-layered model
Document your performance comparison**



Text prediction is somewhat similar to logistic regression in that it is also a multiclass classification problem. The number of classes is equal to the number of unique words in the text i.e. the vocabulary size.

The most fundamental text generation/ predictio model consist of a recurrrent layer that is supposed to promote learnin the relationships between the words. In my model, I started with an embedding layer. This layer is used to create a dense mapping of input word sequences. This is done to compress the feature space so that the following layers (usually LSTM) is able to efficiently capture the word relationships. This was followed by two dense layers that produced the output sequence. 

I have used two baseline models that have comparable architecture. 
Baseline 1 consists of an embedding layer followed by two LSTM layers followed by two dense layers.
baseline 2 consists of an embedding layer followed by maxpooling and convolution layer

On comparing the accuracy of the three models, the accuracy of my model was highest for the input training data despite of the fact that my model was simpler.



**[2 points] Effort to study performance difference when linear activation is used instead of sigmoid (and vice versa)?
How does your performance change when linear activations are used instead of sigmoid, in the last neuron and all other neurons.**



I cannot use linear activation for text generation. Since text generation is a multiclass classification problem.



**[2 points] Effort to evaluate your predictions**

I have evaluated my model by providing all the three models with the same seed text. The sequence of words predicted by each model are in line with the training accuracies seen for those models.

*Seed Data:*


*off toward the madeleine suddenly an object rolled before the duke which he had struck with the toe of his boot it was a large piece of bread spattered with mud then to his amazement monsieur de saulnes saw the duc de hardimont pick up the piece of bread wipe it*


*Next 50 words predicted by Model 1:*


*carefully the reproachful embroidered of the other has something headache before the duke had gave himself i read the cannon of fort henri in the line in vain had you see the reproachful embroidered of the other devil it i have a suspicion and the asylum in one night i/8


*Next 50 words predicted by Model 2:*


*and it his me to me to me would living her baskets and night his and it it white to it to would is i i i i i have have name me i i i i blonde me me me me me myself become their their another i my*


*Next 50 words predicted by Model 3:*


*carefully with his handkerchief embroidered and a continual fit of hunger jeanvictor a gloomy place he had lost a piece of bread spattered with mud three years and as he cried jeanvictor went at last my plate and then unfortunately i was always remembered to feel her warm little hand*


To undertsand the results further, given below are the accuracies for all the models after training them for 100 epochs

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


Staying true to the training accuracy, output of model 2 does not make any sense. It has few meaningful phrases like 'are you astonished' but most of it is just word repreated several times (e.g. me me me me me ).

Model 1 performs better than model 2 in that it doesn't keep repeating words. The model demostrates that it is trying to learn and can spit out different words. However, the words don't make much sense.

My model: In my opinion, model 3 performs slightly better than the other two. It has a couple of phrases that make sense like *'carefully with his handkerchief'* and *'a continual fit of hunger'* or *'he had lost a piece of bread spattered with mud'*. However, there is still scope for improvement.



**[4 points] Effort to code a function that represents your model
After your model is trained, read all the weights, and build your own function/method that serves as the model
Verify that predictions you obtain are same as the one you obtained using your trained model**



Not applicable for for my model since LSTM parameters are complicated. So extracting the parameters and using them to generate a prediction is not feasible. Furthermore, text prediction problem can be seen as multiclass classification problem with the number of classes equal to the size of the vocabulary. This can be enormous; which means that the probability of predicting one word from the vocabulary would be extremely small. This leads to further complications in manual prediction of the next word.



**[4 points] Effort to develop a 'reproducible' code
Someone else must be able to reproduce all your results**



The reproducible Python Notebook can be found at https://github.com/mrinal-r/Experiments/blob/master/Word_Based_Text_Generator.ipynb



**[4 points] Effort to document all the steps above**


This repository contains README.md documenting the project in detail. Link to READMe.md https://github.com/mrinal-r/Experiments/blob/master/README.md

I have also added the original text that was used for training along with the sequences that were generated from the proprocessing step. Everything about the input can be found at https://github.com/mrinal-r/Experiments/tree/master/input-artifacts

Also included in a different folder are trained model parameters for all three models. These were saved using callbacks and checkpoint on the training accuracy. I have also included the tokenizer that containes word to int mappings for all the unique words in the text. Everything about the saved models can be found at https://github.com/mrinal-r/Experiments/tree/master/model-artifacts




