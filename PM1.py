import keras.utils
import numpy as np
import pandas as pd
from keras import Sequential, Model, Input
from keras.src.layers import Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, BatchNormalization, Dropout
import os
from Sub_Functions.Evaluate import main_est_parameters
from Sub_Functions.Load_data import Load_data2, balance2, train_test_split2


def proposed_model_main(x_train,x_test,y_train,y_test,epochs,DB):
    y_train=keras.utils.to_categorical(y_train)
    x_train = np.expand_dims(x_train, axis=-1)  # Shape becomes (samples, 76, 1)
    x_test = np.expand_dims(x_test, axis=-1)
    input_layer = Input(shape=x_train.shape[1:])


    # Encoder
    x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)  # Max Pooling
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x=BatchNormalization()(x)
    x=Dropout(0.3)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)  # Bottleneck layer
    # decoder
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    dense_layer = Dense(16, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(dense_layer)
    # Build autoencoder model
    AE = Model(inputs=input_layer, outputs=output_layer)
    AE.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    AE.fit(x_train,y_train, epochs=epochs, batch_size=8,validation_split=0.2)
    # Compile the model

    # Predict and evaluate
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    preds = AE.predict(x_test)
    pred = np.argmax(preds, axis=1)
    # Placeholder for evaluation metric function
    metrics = main_est_parameters(y_true, pred)
    os.makedirs(f"../Threshold/{DB}",exist_ok=True)
    np.save(f"../Threshold/{DB}/metrics_stored",metrics)

    # threshold_metrics=np.load("../Threshold/metrics_stored.npy")
    #
    # if metrics[]>
    #
    return metrics

feat,labels=Load_data2("CICIDS2015")
balanced_feat,balanced_label=balance2("CICIDS2015",feat,labels)
x_train,x_test,y_train,y_test=train_test_split2(balanced_feat,balanced_label,percent=70)
metrics=proposed_model_main(x_train,x_test,y_train,y_test,epochs=30,DB="CICIDS2015")
print(metrics)