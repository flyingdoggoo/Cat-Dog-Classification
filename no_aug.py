import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

train_dir = "cats_and_dogs_filtered/train"
valid_dir = "cats_and_dogs_filtered/validation"

dog_train_path = 'cats_and_dogs_filtered/train/dogs'
cat_train_path = 'cats_and_dogs_filtered/train/cats'
dog_val_path = 'cats_and_dogs_filtered/validation/dogs'
cat_val_path = 'cats_and_dogs_filtered/validation/cats'

print(f"Total cat training files: {len(os.listdir(cat_train_path))}")
print(f"Total dog training files: {len(os.listdir(dog_train_path))}")
print(f"Total cat valid files: {len(os.listdir(dog_val_path))}")
print(f"Total dog valid files: {len(os.listdir(cat_val_path))}")

dog_fname = os.listdir(dog_train_path)
cat_fname = os.listdir(cat_train_path)
def plot_random():
    random_dog_fnames = random.sample(dog_fname, 4)
    random_cat_fnames = random.sample(cat_fname, 4)

    random_dog = [os.path.join(dog_train_path, name) for name in random_dog_fnames]
    random_cat = [os.path.join(cat_train_path, name) for name in random_cat_fnames]
    fig, axs = plt.subplots(2, 4, figsize=(10,5))

    for i, fnames in enumerate(random_cat + random_dog):
        img = plt.imread(fnames)
        axs[i//4,i%4].imshow(img)
        axs[i//4,i%4].axis(False)

    plt.show()
plot_random()
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(150,150,3)),
        tf.keras.layers.Rescaling(1.0/255),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
model = create_model()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
validation_dataset_final = valid_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE)

history = model.fit(
    train_dataset_final,
    epochs=15,
    validation_data=validation_dataset_final,
)

def plot_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax[0].plot(epochs, acc, 'b', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()
    
    ax[1].plot(epochs, loss, 'b', label='Training Loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation Loss')
    ax[1].set_title('Training and validation loss')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    
    plt.show()
    
plot_loss_acc(history)