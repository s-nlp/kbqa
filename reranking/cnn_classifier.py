# pylint: disable=import-error

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

# print("Tensorflow version " + tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# creating 'data flow' for training data from data directory using ImageDataGenerator class
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 2
IMAGE_SIZE = [256, 256]

# data augmentation for train dataset
train_data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_data_flow = train_data_generator.flow_from_directory(
    directory="/workspace/kbqa/subgraph_plots/training",
    classes=["correct", "wrong"],
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)


# creating 'data flow' for validation data from data directory using ImageDataGenerator class

valid_data_generator = ImageDataGenerator(rescale=1.0 / 255)

valid_data_flow = valid_data_generator.flow_from_directory(
    directory="/workspace/kbqa/subgraph_plots/validation",
    classes=["correct", "wrong"],
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)


EPOCHS = 25

LR_START = 0.00001
LR_MAX = 0.00005
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 8
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8


@tf.function
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        learning_rate = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        learning_rate = LR_MAX
    else:
        learning_rate = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return learning_rate


rng = list(range(EPOCHS))
y = [lrfn(x) for x in rng]

plt.plot(rng, y)
plt.savefig("lr_schedule.png", format="PNG")

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    figure = plt.subplot(subplot)
    figure.set_facecolor("#F8F8F8")
    figure.plot(training)
    figure.plot(validation)
    figure.set_title("model " + title)
    figure.set_ylabel(title)
    figure.set_xlabel("epoch")
    figure.legend(["train", "valid."])
    return figure


# pretrained Xception network as basis for our classifier

pretrained_model_1 = tf.keras.applications.Xception(
    weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
)
pretrained_model_1.trainable = True

model_1 = tf.keras.Sequential(
    [
        pretrained_model_1,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model_1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
earlystoping_callback = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True, monitor="val_accuracy", mode="auto"
)
EPOCHS = 25
STEPS_PER_EPOCH = 30 // BATCH_SIZE

HISTORY_1 = model_1.fit(
    train_data_flow,
    shuffle=True,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=valid_data_flow,
    callbacks=[lr_callback, earlystoping_callback],
)

# Display training curves for modified Xception model

model_loss = display_training_curves(
    HISTORY_1.history["loss"], HISTORY_1.history["val_loss"], "loss", 211
)
plt.savefig("model_loss.png", format="PNG")

model_accuracy = display_training_curves(
    HISTORY_1.history["accuracy"], HISTORY_1.history["val_accuracy"], "accuracy", 212
)
plt.savefig("model_accuracy.png", format="PNG")

# Evaluate trained classifier

Xception_loss, Xception_accuracy = model_1.evaluate(valid_data_flow)

print("model accuracy: " + str(Xception_accuracy))

# make predictions

valid_data_flow_2 = valid_data_generator.flow_from_directory(
    directory="/workspace/kbqa/subgraph_plots/validation",
    classes=["correct", "wrong"],
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=True,
    seed=42,
)


images, valid_labels = next(valid_data_flow_2)
valid_pred = model_1.predict(images)

valid_predictions = np.around(valid_pred)


# plot confusion matrix
cm = confusion_matrix(valid_labels, valid_predictions)
print(cm)

# Accuracy, precision and recall scores for validation set
accuracy = accuracy_score(valid_labels, valid_predictions)
precision = precision_score(valid_labels, valid_predictions)
recall = recall_score(valid_labels, valid_predictions)

print("Accuracy score: " + str(accuracy))
print("Precision score: " + str(precision))
print("Recall score: " + str(recall))
