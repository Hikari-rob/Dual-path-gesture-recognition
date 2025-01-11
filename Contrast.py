import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras import layers, Model, optimizers, callbacks, regularizers
from tensorflow.keras.layers import Conv2D, Dense, Input, Concatenate, Add, MaxPooling2D, AlphaDropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import pandas as pd

# 全局随机种子设置
tf.random.set_seed(2)

# 图像尺寸为64x64
img_width, img_height = 64, 64
# 批次大小
batch_size = 32
# 分类类别数
num_classes = 29

# 数据生成器，用于图像预处理以及数据增强
# 为数据增强操作设置随机种子
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    # seed=42  # 设置数据增强的随机种子
)
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
# test_datagen = ImageDataGenerator(rescale=1. / 255, seed=42)  # 同样设置测试集数据生成器的随机种子
# val_datagen = ImageDataGenerator(rescale=1. / 255, seed=42)  # 设置验证集数据生成器的随机种子

# 训练集数据生成器
train_generator = train_datagen.flow_from_directory(
    os.path.join('Dataset', 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


test_generator = test_datagen.flow_from_directory(
    os.path.join('Dataset', 'test'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)  # 设置shuffle为False，确保顺序一致，方便后续处理

# 验证集数据生成器
val_generator = val_datagen.flow_from_directory(
    os.path.join('Dataset', 'val'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# 模型A
def build_model_A():
    # 输入层尺寸调整为64x64
    # 输入层
    input_layer = layers.Input(shape=(img_width, img_height, 3))

    x1 = layers.Conv2D(32, (3, 3), activation='relu', name="path1_conv1", padding='same')(input_layer)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', name="path2_conv1", padding='same')(input_layer)
    x1 = layers.Add()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv2", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_conv2", padding='same')(x2)
    x1 = layers.Add()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv3", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_conv3", padding='same')(x2)
    x1 = layers.Add()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv4", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_conv4", padding='same')(x2)
    x1 = layers.Add()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D((1, 1))(x1)

    # 新增一层卷积层
    x = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv5", padding='same')(x1)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((2, 2))(x)

    x = layers.Flatten()(x1)
    x = layers.Dense(128, activation='relu')(x)
    x = AlphaDropout(rate=0.5)(x)
    x = layers.Dense(29, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# 模型B
def build_model_B():
    # 输入层尺寸调整为64x64
    input_layer = layers.Input(shape=(img_width, img_height, 3))
    x1 = layers.Conv2D(32, (3, 3), activation='relu', name="path1_c1", padding='same')(input_layer)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', name="path2_c1", padding='same')(input_layer)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_c2", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_c2", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_c3", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_c3", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(32, (3, 3), activation='relu', name="path1_c4", padding='same')(x1)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', name="path2_c4", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    # 池化窗口改为 (1, 1)
    x1 = layers.AveragePooling2D((1, 1))(x1)

    # 新增一层卷积层
    x = layers.Conv2D(64, (3, 3), activation='relu', name="path1_c5", padding='same')(x1)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = AlphaDropout(rate=0.5)(x)
    x = layers.Dense(29, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# 用于存储每个模型的训练和验证历史信息
histories = []
# 模型名称列表，方便后续区分展示
model_names = ["Model A", "Model B"]
# 循环构建、训练和评估模型A和模型B
for build_model_func in [build_model_A, build_model_B]:
    # 创建模型实例
    model = build_model_func()

    # 编译模型，设置学习率为0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 在编译模型之后、训练模型之前添加以下代码来输出模型架构信息
    model.summary()
    # 训练模型，设置训练轮次为30，并添加早停回调
    epochs = 30
    # 早停回调设置，监控验证集准确率，当连续5轮验证集准确率不再提升时停止训练
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[early_stopping])
    histories.append(history)

    # 在测试集上评估模型，获取预测结果和真实标签等指标
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f'{model_names[len(histories) - 1]} Test accuracy:', test_acc)

    y_true = []
    for i in range(len(test_generator)):
        batch_labels = test_generator[i][1]
        y_true.extend(np.argmax(batch_labels, axis=1))
    y_true = np.array(y_true)

    y_pred = []
    for i in range(len(test_generator)):
        batch_pred = model.predict(test_generator[i][0])
        y_pred.extend(np.argmax(batch_pred, axis=1))
    y_pred = np.array(y_pred)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'{model_names[len(histories) - 1]} Test Precision:', precision)
    print(f'{model_names[len(histories) - 1]} Test Recall:', recall)
    print(f'{model_names[len(histories) - 1]} Test F1 - score:', f1)

    # 绘制混淆矩阵
    confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
    confusion_matrix = confusion_matrix.numpy()

    df_cm = pd.DataFrame(confusion_matrix, index=train_generator.class_indices.keys(),
                         columns=train_generator.class_indices.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_names[len(histories) - 1]} Confusion Matrix')
    plt.show()

# 可视化展示每个模型训练过程中的准确率和损失变化
for i in range(len(histories)):
    plt.plot(histories[i].history['accuracy'], label=f'{model_names[i]} accuracy')
    plt.plot(histories[i].history['val_accuracy'], label=f'{model_names[i]} val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title(f'{model_names[i]} Training and Validation Accuracy')
    plt.show()

    plt.plot(histories[i].history['loss'], label=f'{model_names[i]} loss')
    plt.plot(histories[i].history['val_loss'], label=f'{model_names[i]} val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_names[i]} Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.show()