"""
https://www.tensorflow.org/tutorials/images/intro_to_cnns?hl=ja
に載ってるサンプルをちょっと改変

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.backend import learning_phase
from model import get_model

# ================
# 手動設定パラメータ
# ================
model_name = 'cnn' # 'cnn'か'resnet50' か 'my_cnn' (model.pyを参照)
epoch = 25
learning_rate = 0.0001
batch_size = 16














# ===================
# データセットの読み込み
# ===================
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data() # サイズは 28x28のモノクロ画像


# ===================
# データセットの前処理
# ===================
train_images = train_images.reshape((60000, 28, 28, 1)) # 28x28x1の配列にする
test_images = test_images.reshape((10000, 28, 28, 1)) # 28x28x1の配列にする
if model_name == 'resnet50':
    # ResNet50というモデルは32x32以上の画像サイズしか受け取れないため、リサイズする
    train_images = tf.image.resize(train_images, [32,32])
    test_images = tf.image.resize(test_images, [32,32])




# ===================
# モデルの読み込み
# ===================
model = get_model(model_name=model_name, input_shape=train_images.shape[1:])
adam = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
print(model.summary())



# ===================
# モデルを学習
# ===================
history = model.fit(train_images, train_labels, epochs=epoch, validation_split=0.2, verbose=2)



# ============
# モデルをテスト
# ============
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) #予測結果から正解率まで自動計算



print('test_loss', test_loss)
print('test_accuracy', test_acc)


# =================
# 学習結果のグラフを保存
# =================
# Loss
plt.figure(figsize=(8,8))
plt.title('Loss Value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlim(0, epoch)
plt.show()
plt.savefig('loss_history.png')
plt.clf()

# Accuracy
plt.figure(figsize=(8,8))
plt.title('Accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.xlim(0, epoch)
plt.ylim(0, 1.0)
plt.show()
plt.savefig('accuracy_history.png')
plt.clf()








"""
追加
"""
tf.keras.models.save_model(model, 'model.h5')
predictions = model.predict(test_images) #予測結果のみ取得
print(predictions[0]) #テストデータセットの1枚目の予測結果。(各数字である確率)
print(predictions[0].argmax()) #テストデータセットの1枚目の予測結果。(何の数字か。)
"""
追加ここまで
"""
