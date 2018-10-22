# mini_project_2
This is mini project 2, which consists of three trained models. 

Model_vgg.py use CNN based on vgg model to train a classifying model. 

Model_softmax.py uses softmax function to train a model.

Model_MLP.py uses Multilayer Perceptron to train a model.

Each model will output an .h5 file which can be use in the next step.

Use test_predict.py to predict a photo, by doing this you must have a test photo under current folder.

The dataset contains of 2 classes -- daisy & tulips, so all of the models could only classify either one of them.


Results:
The Model using vgg layers has done a good job better than any other model, the accuracy of it is approximately 90%.
Model using Multilayer Perceptron also works well, and it has an accuracy about 75%.
However, Model using softmax has an accuracy about 50%, which means it has little effect in binary classify task.
