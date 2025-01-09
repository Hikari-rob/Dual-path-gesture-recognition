import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras import layers, Model, optimizers, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping


# 随机种子
tf.random.set_seed(2)

# 图像尺寸，增大为64x64
img_width, img_height = 64, 64
# 批次大小
batch_size = 32
# 分类类别数（假设这里是识别A - Z+3个类共29个字母）
num_classes = 29

# 数据生成器，用于图像预处理以及数据增强（这里可以根据需求调整增强参数）
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练集数据生成器
train_generator = train_datagen.flow_from_directory(
    os.path.join('Dataset', 'train'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# 确保测试集数据生成器生成完整的批次，避免样本数量不一致问题
test_generator = test_datagen.flow_from_directory(
    os.path.join('Dataset', 'test'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# 验证集数据生成器
val_generator = val_datagen.flow_from_directory(
    os.path.join('Dataset', 'val'),
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# 构建类似CNN模型
def build_model():
    # 输入层尺寸调整为64x64
    input_layer = layers.Input(shape=(img_width, img_height, 3))
    x1 = layers.Conv2D(32, (3, 3), activation='relu', name="path1_conv1", padding='same')(input_layer)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', name="path2_conv1", padding='same')(input_layer)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv2", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_conv2", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv3", padding='same')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', name="path2_conv3", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.AveragePooling2D((2, 2))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x1 = layers.Conv2D(32, (3, 3), activation='relu', name="path1_conv4", padding='same')(x1)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', name="path2_conv4", padding='same')(x2)
    x1 = layers.Multiply()([x1, x2])
    x1 = layers.BatchNormalization()(x1)
    # 池化窗口改为 (1, 1)，减缓维度下降速度
    x1 = layers.AveragePooling2D((1, 1))(x1)

    # 新增一层卷积层
    x = layers.Conv2D(64, (3, 3), activation='relu', name="path1_conv5", padding='same')(x1)
    x = layers.AveragePooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(29, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# 创建模型实例
model = build_model()

# 编译模型，设置学习率为0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 在编译模型之后、训练模型之前添加以下代码来输出模型架构信息
model.summary()
# 训练模型，设置训练轮次为50，这里将早停回调相关代码注释掉，模型会训练完50轮次
epochs = 30
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping])
# 可视化展示训练过程中的准确率和损失变化
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.show()

# 在测试集上评估模型，获取预测结果和真实标签
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# 获取测试集的真实标签（确保获取完整且准确的标签）
y_true = []
for i in range(len(test_generator)):
    batch_labels = test_generator[i][1]
    y_true.extend(np.argmax(batch_labels, axis=1))
y_true = np.array(y_true)

# 获取预测结果（确保与真实标签顺序对应且数量一致）
y_pred = []
for i in range(len(test_generator)):
    batch_pred = model.predict(test_generator[i][0])
    y_pred.extend(np.argmax(batch_pred, axis=1))
y_pred = np.array(y_pred)

# 计算精准率、召回率、F1分数（针对多分类任务，采用宏平均方式，可根据需求调整为其他平均方式）
precision = metrics.precision_score(y_true, y_pred, average='macro')
recall = metrics.recall_score(y_true, y_pred, average='macro')
f1 = metrics.f1_score(y_true, y_pred, average='macro')

print('Test Precision:', precision)
print('Test Recall:', recall)
print('Test F1 - score:', f1)

# 绘制混淆矩阵
confusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
confusion_matrix = confusion_matrix.numpy()

# 将混淆矩阵转换为DataFrame格式，方便标注行列标签
df_cm = pd.DataFrame(confusion_matrix, index=train_generator.class_indices.keys(),
                     columns=train_generator.class_indices.keys())

# 使用seaborn绘制混淆矩阵热图
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 保存模型
model.save('trained_model_cnn_3.h5')

# 整理数据以输出表格
categories = train_generator.class_indices.keys()
precision_scores = []
recall_scores = []
f1_scores = []
num_samples = []

for category in categories:
    category_index = list(train_generator.class_indices.keys()).index(category)
    true_positives = confusion_matrix[category_index][category_index]
    false_positives = np.sum(confusion_matrix[:, category_index]) - true_positives
    false_negatives = np.sum(confusion_matrix[category_index, :]) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    num_sample = np.sum(confusion_matrix[category_index, :])

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_score)
    num_samples.append(num_sample)

# 创建DataFrame并输出表格
data = {
    '类别': categories,
    '精确率': precision_scores,
    '召回率': recall_scores,
    'F1 - Score': f1_scores,
    '实际类别数目': num_samples
}
df = pd.DataFrame(data)

# 添加图片中表格最后三行内容
last_rows = [
    ['Micro', metrics.precision_score(y_true, y_pred, average='micro'), metrics.recall_score(y_true, y_pred, average='micro'),
     metrics.f1_score(y_true, y_pred, average='micro'), len(y_true)],
    ['Macro', metrics.precision_score(y_true, y_pred, average='macro'), metrics.recall_score(y_true, y_pred, average='macro'),
     metrics.f1_score(y_true, y_pred, average='macro'), len(y_true)],
    ['Weighted', metrics.precision_score(y_true, y_pred, average='weighted'), metrics.recall_score(y_true, y_pred, average='weighted'),
     metrics.f1_score(y_true, y_pred, average='weighted'), len(y_true)]
]
columns = ['类别', '精确率', '召回率', 'F1 - Score', '实际类别数目']
df_last_rows = pd.DataFrame(last_rows, columns=columns)
df = pd.concat([df, df_last_rows])

print(df)