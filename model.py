from tensorflow.keras.models import Sequential, Model
from tensorflow import keras
from tensorflow.keras import layers, models


def get_model(model_name, input_shape):
    if model_name == 'cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        return model




    elif model_name == 'resnet50':
        base_resnet50 = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
        x = base_resnet50.output
        predictions = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=base_resnet50.input, outputs=predictions)
        return model



    if model_name == 'my_cnn':
        """
        自分でモデルを作成するときは、ここを使ってください。
        (実際に学習で使うときはmain.pyのmodel_nameを'my_cnn'に変更してください)
        現時点ではmodel_name == 'cnn'と同じ状態になっています。
        https://keras.io/ja/layers/convolutional/
        https://keras.io/ja/layers/core/
        などで、どんなレイヤーがあるか確認できます。
        このプログラムはchannel_firstで記述されているので、Conv2D(4, (3, 3))は、3x3のカーネルサイズ、畳み込んで4ch出力という意味になります。
        """
        model = models.Sequential()
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        return model