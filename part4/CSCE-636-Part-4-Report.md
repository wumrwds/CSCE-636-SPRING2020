# CSCE-636-Part-4-Report

### Model Design

The main idea of my model is: first, we use CNN to extract the  spatial features of each video frame, and then feed them to the LSTM network to obtain sequence timing features. 



The code goes like the following:

```python
video = Input(shape=(frames, rows, columns, channels))
cnn_base = VGG16(input_shape=(rows, columns, channels), weights="imagenet", include_top=False)
cnn_out = GlobalAveragePooling2D()(cnn_base.output)
cnn = Model(input=cnn_base.input, output=cnn_out)
cnn.trainable = False

encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(256)(encoded_frames)
hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
model = Model([video], outputs)

optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["categorical_accuracy"]) 
```

First, we use a special kind of CNN called VGG16 as the first layer. Then do global average pooling. Later encode the CNN output as the sequences and feed them to the LSTM network. Finally, we use a rely layer and a softmax layer to do the classification.



### Dataset

I used the [CASIA action database for recognition]([http://www.cbsr.ia.ac.cn/english/Action%20Databases%20EN.asp](http://www.cbsr.ia.ac.cn/english/Action Databases EN.asp)) as the training dataset and then I recorded some short clips of videos by myself which I have uploaded on my [Google Drive](https://drive.google.com/drive/folders/1uKkHqvemRo8D1EMF4H_iPHl0Hnjo3jYA?usp=sharing) as the testing dataset.



CASIA dataset contains eight different types of actions of single person from 3 different perspectives, such as walking, running, bending, jumping, crouching, fainting, wandering and punching a car.

<img src="/Users/minrengwu/Git/github/CSCE-636-SPRING2020/part4/pics/CASIA_action_database_single.jpg" alt="img" style="zoom:50%;" />



### Skeleton Landmarks

 We can use the open-source project [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to add skeleton landmarks for our videos so that we can get a better prediction accuracy.

I compiled this project on my computer and processed all the videos in my dataset.



A sample of the training dataset:

![image-20200326000623756](/Users/minrengwu/Git/github/CSCE-636-SPRING2020/part4/pics/image-20200326000623756.png)



A sample of the testing dataset:

![image-20200326000726299](/Users/minrengwu/Git/github/CSCE-636-SPRING2020/part4/pics/image-20200326000726299.png)



### Run the Code

I separate my model into two scripts: one is `model_train.py`, the other is `model_evaluate.py`.

In `model_train.py`, I pre-processed the training set and train the model. After successfully training the model, I save the model as a h5 file `my_model.h5`.

In  `model_evaluate.py`, we can use this script to do the prediction. For example, if we want to predict which category the file `run_test_1.mp4` is in, we can do the prediction by running the  `model_evaluate.py` script:

```bash
$ python model_evaluate.py dataset/test/run/run_test_1.mp4
```



Then, we will get the result like the following:

![image-20200326004634509](/Users/minrengwu/Git/github/CSCE-636-SPRING2020/part4/pics/image-20200326004634509.png)