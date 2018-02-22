import keras
from keras.optimizers import SGD
from keras.models import model_from_json
import json
import numpy as np

test_X=np.load("test_X.npy")
ids=np.load("ids.npy")

json_file = open("Model_all.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights_all.h5")
sgd=SGD(lr=0.01,decay=1e-6,nesterov=True)
model.compile(optimizer=sgd,loss="binary_crossentropy",metrics=["accuracy"])
count=1
with open('submission_all.csv', 'w') as f:
    f.write('connection_id,target\n')
    for i in range(len(test_X)):

        features=test_X[i]
        pred=model.predict(np.array([features]))
        lbl = np.argmax(pred[0,:])
        f.write('{},{}\n'.format(ids[i],lbl))
        if count %10000 is 0:
            print(count)
        count+=1