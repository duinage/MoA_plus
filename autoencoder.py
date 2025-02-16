import numpy as np
import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Reshape
# from tensorflow.keras.models import Model
# from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt

# # Load and preprocess MNIST data
# (x_train, _), (x_test, _) = mnist.load_data()

# # Normalize and flatten data
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))

# # Autoencoder architecture


# # Compile the autoencoder
# autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# # Train the autoencoder
# history = autoencoder.fit(
#     x_train, x_train,
#     epochs=50,
#     batch_size=256,
#     shuffle=True,
#     validation_data=(x_test, x_test)
# )

# # Visualize training history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Autoencoder Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Encode and decode some digits
# encoded_imgs = autoencoder.predict(x_test)
# decoded_imgs = autoencoder.predict(x_test)

# # Visualize original and reconstructed images
# n = 10  # Number of digits to display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # Original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
#     plt.title("Original")
#     plt.axis("off")

#     # Reconstructed
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
#     plt.title("Reconstructed")
#     plt.axis("off")
# plt.show()