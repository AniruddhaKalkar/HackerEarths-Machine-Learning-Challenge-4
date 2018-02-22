import keras
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,Adam
from keras.models import Sequential
import numpy as np
import math
X=np.load("train_X.npy")
y=np.load("train_y.npy")
test_percent=math.ceil(0.1*len(X))
X_train=X[:-test_percent]
y_train=y[:-test_percent]
X_test=X[-test_percent:]
y_test=y[-test_percent:]

input_nodes=X.shape[1]
hl1_nodes=1000
hl2_nodes=5000
hl3_nodes=1000
output_nodes=y.shape[1]
batch_size=128
epochs=5
model=Sequential()
model.add(Dense(units=input_nodes,input_dim=input_nodes,activation="relu"))
model.add(Dense(units=hl1_nodes,activation="relu"))
model.add(Dropout(0.8))
model.add(Dense(units=hl2_nodes,activation="relu"))
model.add(Dropout(0.8))
model.add(Dense(units=hl3_nodes,activation="relu"))
model.add(Dense(units=output_nodes,activation="softmax"))
sgd=SGD(lr=0.01,decay=1e-6,nesterov=True)
ad=Adam(lr=0.001,decay=1e-6)
model.compile(optimizer=ad,loss="binary_crossentropy",metrics=["accuracy"])
print(model.summary())
model.fit(X,y,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
expnense_model=model.to_json()
with open("Model_all.json","w") as json_file:
    json_file.write(expnense_model)
model.save_weights("weights_all.h5")
