import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras_radam import RAdam


# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())


def split_X(X):
    X_riasec = X[:, :48]
    X_riasec_meta = X[:, 48:51]
    X_tipi = X[:, 51:61]
    X_vcl = X[:, 61:77]
    X_education = X[:, 77:78]
    X_age = X[:, 78:79]
    X_family_size = X[:, 79:80]
    X_urban = X[:, 80:84]
    X_gender = X[:, 84:88]
    X_eng_nat = X[:, 88:91]
    X_hand = X[:, 91:95]
    X_religion = X[:, 95:108]
    X_orientation = X[:, 108:114]
    X_race = X[:, 114:120]
    X_voted = X[:, 120:123]
    X_married = X[:, 123:127]
    X_net_loc = X[:, 127:129]
    X_country = X[:, 129:316]
    X_source = X[:, 316:319]

    return [
        X_riasec, X_riasec_meta, X_tipi, X_vcl, X_education, X_age,
        X_family_size, X_urban, X_gender, X_eng_nat, X_hand, X_religion,
        X_orientation, X_race, X_voted, X_married, X_net_loc, X_country,
        X_source
    ]


def build_input_model():
    riasec = keras.Input(shape=(48,), name='RIASEC_IN')
    riasec_meta = keras.Input(shape=(3,), name='RIASEC_META_IN')
    tipi = keras.Input(shape=(10,), name='TIPI_IN')
    vcl = keras.Input(shape=(16,), name='VCL_IN')
    net_loc = keras.Input(shape=(2,), name='NET_LOC_IN')
    country = keras.Input(shape=(187,), name='COUNTRY_IN')
    source = keras.Input(shape=(3,), name='SOURCE_IN')

    net_loc_enc = keras.layers.Dense(1, activation='relu', name='NET_LOC_ENC')(
        net_loc)
    country_enc = keras.layers.Dense(1, activation='relu', name='COUNTRY_ENC')(
        country)
    source_enc = keras.layers.Dense(1, activation='relu', name='SOURCE_ENC')(
        source)

    final_input_layer = keras.layers.Concatenate(axis=1, name='FINAL_INPUT')([
        riasec, riasec_meta, tipi, vcl, net_loc_enc, country_enc, source_enc
    ])


    return keras.models.Model(
        inputs=[
            riasec, riasec_meta, tipi, vcl, net_loc, country, source
        ],
        outputs=final_input_layer

    )


def build_model():

    def custom_block(neurons=500, drop=True, name=None):

        if name:
            dense_name = name+'_DENSE'
            relu_name = name + '_RELU'
            bn_name = name + '_BN'
            drop_name = name + '_DROPOUT'
        else:
            relu_name = None
            bn_name = None
            drop_name = None

        def _custom_block(x):
            x = keras.layers.Dense(neurons, name=dense_name)(x)
            x = keras.layers.LeakyReLU(name=relu_name)(x)
            x = keras.layers.BatchNormalization(name=bn_name)(x)
            if drop:
                x = keras.layers.Dropout(0.5, name=drop_name)(x)

            return x

        return _custom_block

    input_model = build_input_model()
    x = keras.layers.Dropout(0.5, name='INPUT_DROPOUT')(input_model.output)
    x1 = custom_block(name='X1')(x)
    x2 = custom_block(name='X2')(x1)
    x12 = keras.layers.Concatenate(axis=-1, name='RESIDUAL')([x1, x2])
    bottleneck = custom_block(10, drop=False, name='BOTTLENECK')(x12)
    # x3 = custom_block(100)(bottleneck)
    '''
    output_layer = keras.layers.Dense(319, activation='sigmoid', name='OUTPUT')(
        bottleneck)
    '''

    '''
    hidden_model = keras.models.Model(inputs=input_model.input,
        outputs=bottleneck)
    output_layer = build_output(hidden_model)
    '''

    education = keras.layers.Dense(1, activation='sigmoid', name='EDUCATION_OUT'
        )(bottleneck)
    age = keras.layers.Dense(1, activation='sigmoid', name='AGE_OUT')(bottleneck
        )
    family_size = keras.layers.Dense(1, activation='sigmoid',
        name='FAMILY_SIZE_OUT')(bottleneck)
    urban = keras.layers.Dense(4, activation='sigmoid', name='URBAN_OUT')(
        bottleneck)
    gender = keras.layers.Dense(4, activation='sigmoid', name='GENDER_OUT')(
        bottleneck)
    eng_nat = keras.layers.Dense(3, activation='sigmoid', name='ENG_NAT_OUT')(
        bottleneck)
    hand = keras.layers.Dense(4, activation='sigmoid', name='HAND_OUT')(
        bottleneck)
    religion = keras.layers.Dense(13, activation='sigmoid', name='RELIGION_OUT'
        )(bottleneck)
    orientation = keras.layers.Dense(6, activation='sigmoid',
        name='ORIENTATION_OUT')(bottleneck)
    race = keras.layers.Dense(6, activation='sigmoid', name='RACE_OUT')(
        bottleneck)
    voted = keras.layers.Dense(3, activation='sigmoid', name='VOTED_OUT')(
        bottleneck)
    married = keras.layers.Dense(4, activation='sigmoid', name='MARRIED_OUT')(
        bottleneck)

    return keras.models.Model(inputs=input_model.input, outputs=[
        education, age, family_size, urban, gender, eng_nat, hand, religion, orientation, race,
        voted, married
    ])


df = pd.read_csv("ppData.csv")
X = df.values  # returns a numpy array

# chose min_max_scaler in order to preserve values of 0
# important because nans were replaced with 0
min_max_scaler = preprocessing.MinMaxScaler()
# min_max_scaler = preprocessing.StandardScaler()
X = min_max_scaler.fit_transform(X)

# inverse = min_max_scaler.inverse_transform(normalized)
'''
df = pd.DataFrame(X)
df.to_csv('ppData2.csv', index=False)
'''

# Y = X[:, 79:127]
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=15000,
    random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
    test_size=15000, random_state=42)
'''

X_train, X_test, dummy1, dummy2 = train_test_split(X, X, test_size=15000,
    random_state=42)
X_train, X_val, dummy1, dummy2 = train_test_split(X_train, X_train,
    test_size=15000, random_state=42)

print(X.shape)
# print(Y.shape)
print(X_train.shape)
# print(Y_train.shape)
print(X_val.shape)
# print(Y_val.shape)
print(X_test.shape)
# print(Y_test.shape)

'''
(
    X_train_riasec, X_train_riasec_meta, X_train_tipi, X_train_vcl,
    X_train_education, X_train_age, X_train_family_size, X_train_urban,
    X_train_gender, X_train_eng_nat, X_train_hand, X_train_religion,
    X_train_orientation, X_train_race, X_train_voted, X_train_married,
    X_train_net_loc, X_train_country, X_train_source
) = split_X(X_train)

(
    X_test_riasec, X_test_riasec_meta, X_test_tipi, X_test_vcl,
    X_test_education, X_test_age, X_test_family_size, X_test_urban,
    X_test_gender, X_test_eng_nat, X_test_hand, X_test_religion,
    X_test_orientation, X_test_race, X_test_voted, X_test_married,
    X_test_net_loc, X_test_country, X_test_source
) = split_X(X_test)

(
    X_val_riasec, X_val_riasec_meta, X_val_tipi, X_val_vcl, X_val_education,
    X_val_age, X_val_family_size, X_val_urban, X_val_gender, X_val_eng_nat,
    X_val_hand, X_val_religion, X_val_orientation, X_val_race, X_val_voted,
    X_val_married, X_val_net_loc, X_val_country, X_val_source
) = split_X(X_val)
'''

X_train = split_X(X_train)
X_test = split_X(X_test)
X_val = split_X(X_val)

X2_train = X_train[:4] + X_train[16:]
Y2_train = X_train[4:16]

X2_test = X_test[:4] + X_test[16:]
Y2_test = X_test[4:16]

X2_val = X_val[:4] + X_val[16:]
Y2_val = X_val[4:16]

print(len(X_train))
print(len(X2_train))
print(len(Y2_train))

model = build_model()
print(model.summary())
keras.utils.plot_model(model, to_file='personality2vec2.png', show_shapes=True,
    show_layer_names=True)

model.load_weights('personality2vec.h5', by_name=True, skip_mismatch=True)

losses = {
    "EDUCATION_OUT": "mean_squared_error",
    "AGE_OUT": "mean_squared_error",
    "FAMILY_SIZE_OUT": "mean_squared_error",
    "URBAN_OUT": "categorical_crossentropy",
    "GENDER_OUT": "categorical_crossentropy",
    "ENG_NAT_OUT": "categorical_crossentropy",
    "HAND_OUT": "categorical_crossentropy",
    "RELIGION_OUT": "categorical_crossentropy",
    "ORIENTATION_OUT": "categorical_crossentropy",
    "RACE_OUT": "categorical_crossentropy",
    "VOTED_OUT": "categorical_crossentropy",
    "MARRIED_OUT": "categorical_crossentropy"
}
model.compile(RAdam(), loss=losses)
# ,metrics=["accuracy"]

EPOCHS = 100
BS = 64

checkpoint = keras.callbacks.ModelCheckpoint(
    '/home/gabriel/Documents/Fun/Python Projects/PersonalityEncoding/'
    + 'saved-model2-{epoch:02d}-{val_loss:.2f}.hdf5',
     monitor='val_loss', verbose=0, save_best_only=False,
     save_weights_only=True, mode='auto', period=5
)
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
    write_graph=True, write_images=True)
# tensorboard --logdir Graph/
callbacks = [checkpoint, tensorboard]

model.fit(
    x=X2_train,
    y=Y2_train,
    batch_size=BS,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=(X2_val, Y2_val),
    shuffle=True,
    verbose=1)

# no labels as input
# fine-tuned vs raw

# decide Y
# baseline network
# test performance

model.save_weights('personality2vec2.h5')
# model.load_weights('personality2vec.h5')

Y_pred = model.predict_classes(X_test)