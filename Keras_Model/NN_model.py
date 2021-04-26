from cleaning_dataset import *


model=keras.Sequential([
    keras.layers.Dense(1,input_shape=(2,),activation='sigmoid',kernel_initializer='ones',bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled,y_train,epochs=5000)

model.save("model.h5")
#print("")
#print("The accuracy of model on train dataset is 91%")
#print("")

model.evaluate(X_test_scaled,y_test)

#print("")
#print("The performance of the model is 100%")
#print("")

