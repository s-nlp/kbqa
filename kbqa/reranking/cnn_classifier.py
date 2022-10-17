import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch_size",
    default=2,
    type=int,
)

parser.add_argument(
    "--train_epochs",
    default=15,
    type=int,
)

parser.add_argument(
    "--directory",
    default="/workspace/kbqa/subgraph_plots/",
    type=str,
)

parser.add_argument(
    "--test_samples",
    default=50,
    type=int,
)


def data_generator(directory, batch_size):

    """
    function to load data
    """
    data_flow = train_data_generator.flow_from_directory(
        directory=directory,
        classes=["correct", "wrong"],
        target_size=[256, 256],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42,
    )

    return data_flow


class LearningRate:

    """
    Class to initialize and define learning rate
    """

    def __init__(self):
        self.lr_start = 0.00001
        self.lr_max = 0.00005
        self.lr_min = 0.00001
        self.lr_rampup_epochs = 8
        self.lr_sustain_epochs = 0
        self.lr_exp_decay = 0.8

    @tf.function
    def lrfn(self, epoch):
        """
        function to define our variable learning rate
        """

        if epoch < self.lr_rampup_epochs:
            learning_rate = (
                self.lr_max - self.lr_start
            ) / self.lr_rampup_epochs * epoch + self.lr_start
        elif epoch < self.lr_rampup_epochs + self.lr_sustain_epochs:
            learning_rate = self.lr_max
        else:
            learning_rate = (self.lr_max - self.lr_min) * self.lr_exp_decay ** (
                epoch - self.lr_rampup_epochs - self.lr_sustain_epochs
            ) + self.lr_min

        return learning_rate

    def l_r(self, epochs):

        """
        function to initialize learning rate
        """
        rng = list(range(epochs))
        y_axs = [self.lrfn(x) for x in rng]

        return rng, y_axs


def model(image_size):

    """
    function to build model using pre trained network with added layer for binary classification
    """
    # pretrained Xception network as basis for our classifier
    pretrained_model = tf.keras.applications.Xception(
        weights="imagenet", include_top=False, input_shape=[*image_size, 3]
    )
    pretrained_model.trainable = True

    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def display_training_curves(training, validation, title, subplot):

    """
    function to plot training curves
    """
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


def metrics(valid_labels, valid_predictions):

    """
    function to evaluate metrics on validation
    """
    # confusion matrix
    conf_mat = confusion_matrix(valid_labels, valid_predictions)

    # Accuracy, precision and recall scores for validation set
    accuracy = accuracy_score(valid_labels, valid_predictions)
    precision = precision_score(valid_labels, valid_predictions)
    recall = recall_score(valid_labels, valid_predictions)

    return conf_mat, accuracy, precision, recall


if __name__ == "__main__":
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = [256, 256]

    # creating 'data flow' for training data from data directory using ImageDataGenerator class
    # data augmentation for train dataset
    train_data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    train_data_flow = data_generator(args.directory + "training", BATCH_SIZE)

    # creating 'data flow' for validation data from data directory using ImageDataGenerator class
    valid_data_generator = ImageDataGenerator(rescale=1.0 / 255)

    valid_data_flow = data_generator(args.directory + "validation", BATCH_SIZE)

    EPOCHS = args.train_epochs

    learning_r = LearningRate()

    rng, y = learning_r.l_r(EPOCHS)

    plt.plot(rng, y)
    plt.savefig("lr_schedule.png", format="PNG")

    print(
        "Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1])
    )

    model = model(IMAGE_SIZE)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        learning_r.lrfn, verbose=True
    )
    earlystoping_callback = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor="val_accuracy", mode="auto"
    )

    STEPS_PER_EPOCH = 30 // BATCH_SIZE

    HISTORY = model.fit(
        train_data_flow,
        shuffle=True,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=valid_data_flow,
        callbacks=[lr_callback, earlystoping_callback],
    )

    # Display training curves for modified Xception model
    model_loss = display_training_curves(
        HISTORY.history["loss"], HISTORY.history["val_loss"], "loss", 211
    )
    plt.savefig("model_loss.png", format="PNG")

    model_accuracy = display_training_curves(
        HISTORY.history["accuracy"], HISTORY.history["val_accuracy"], "accuracy", 212
    )
    plt.savefig("model_accuracy.png", format="PNG")

    # Evaluate trained classifier
    Xception_loss, Xception_accuracy = model.evaluate(valid_data_flow)

    print("model accuracy: " + str(Xception_accuracy))

    # make predictions
    test_data = data_generator(args.directory + "validation", args.test_samples)

    images, valid_labels = next(test_data)
    valid_pred = model.predict(images)

    valid_predictions = np.around(valid_pred)

    cm, accuracy, precision, recall = metrics(valid_labels, valid_predictions)

    # print confusion matrix
    print(cm)

    # print metrics
    print("Accuracy score: " + str(accuracy))
    print("Precision score: " + str(precision))
    print("Recall score: " + str(recall))
