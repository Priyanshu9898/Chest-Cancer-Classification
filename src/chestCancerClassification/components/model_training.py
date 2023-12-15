import tensorflow as tf
from pathlib import Path
from chestCancerClassification.entity import TrainingConfig
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import pandas as pd
from chestCancerClassification import logger
from sklearn.metrics import precision_recall_fscore_support
import mlflow
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


class Training:
    def __init__(self, config: TrainingConfig, model_name: str):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.model_name = model_name

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def setup_data_generators(self):
        datagenerator_kwargs = dict(rescale=1./255)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Train generator with augmentation
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40, horizontal_flip=True, width_shift_range=0.2, 
            height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
            **datagenerator_kwargs
        ) if self.config.params_is_augmentation else tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data, shuffle=True, **dataflow_kwargs
        )

        # Validation generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data, shuffle=False, **dataflow_kwargs
        )

        # Test generator
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.test_data, shuffle=False, **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Training accuracy')
        plt.plot(epochs, val_acc, label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.savefig(os.path.join(self.config.results_folder, f'{self.model_name}_training_history.png'))
        # plt.show()

    # def evaluate_and_report(self):
    #     test_loss, test_accuracy = self.model.evaluate(self.test_generator)
    #     print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

    #     predictions = np.argmax(self.model.predict(self.test_generator), axis=1)
    #     true_classes = self.test_generator.classes
    #     class_labels = list(self.test_generator.class_indices.keys())

    #     cm = confusion_matrix(true_classes, predictions)
    #     self.plot_confusion_matrix(cm, class_labels)

    #     precision, recall, f1_score, _ = precision_recall_fscore_support(true_classes, predictions, average='weighted')
    #     self.append_to_csv(test_accuracy, precision, recall, f1_score)

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.config.results_folder, f'confusion_matrix_{self.model_name}.png'))
        # plt.show() 

    def append_to_csv(self, accuracy, precision, recall, f1_score):
        results_file = os.path.join(self.config.results_folder, 'results.csv')
        new_row = {'Model': self.model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}

        if not os.path.isfile(results_file):
            pd.DataFrame([new_row]).to_csv(results_file, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(results_file, mode='a', header=False, index=False)

    def train(self):
        # create_directories([self.config.results_folder])

        self.get_base_model()
        self.setup_data_generators()

        steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
         # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)


        with mlflow.start_run(run_name=self.model_name):
            # Log parameters
            mlflow.log_param("image_size", self.config.params_image_size)
            mlflow.log_param("batch_size", self.config.params_batch_size)
            mlflow.log_param("epochs", self.config.params_epochs)

            # Train model
            history = self.model.fit(
                self.train_generator, 
                epochs=self.config.params_epochs, 
                steps_per_epoch=steps_per_epoch, 
                validation_data=self.valid_generator, 
                validation_steps=validation_steps,
                callbacks=[reduce_lr, early_stop]
            )

            # Log training metrics
            mlflow.log_metric("train_accuracy", max(history.history['accuracy']))
            mlflow.log_metric("train_loss", min(history.history['loss']))

            self.save_model(path=self.config.trained_model_path, model=self.model)
            self.plot_training_history(history)

            self.evaluate_and_report(history)

    def evaluate_and_report(self, history):
        test_loss, test_accuracy = self.model.evaluate(self.test_generator)
        predictions = np.argmax(self.model.predict(self.test_generator), axis=1)
        true_classes = self.test_generator.classes

        precision, recall, f1_score, _ = precision_recall_fscore_support(true_classes, predictions, average='weighted')

        # Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1_score)

       
        mlflow.keras.log_model(self.model, "model")

        cm = confusion_matrix(true_classes, predictions)
        self.plot_confusion_matrix(cm, list(self.test_generator.class_indices.keys()))

        self.append_to_csv(test_accuracy, precision, recall, f1_score)