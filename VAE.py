import streamlit as st
import os
import numpy as np
import tensorflow as tf

# Define a function to generate a VAE image model
def load_image():
    loaded_model = tf.keras.models.load_model('VAE_MODEL.keras')
    num_samples = 1
    latent_dim = 100
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_images = loaded_model.predict(noise)
    return (generated_images[0] * 127.5 + 127.5).astype(np.uint8)

def main():
    st.title('Load VAE Image')
    if st.button('Load VAE Image'):
        # Get Image from load_image function after loaing VAE loade
        random_img = load_image()
        if random_img is not None:
            # Display the image
            st.image(random_img, caption='Load VAE Image', use_column_width=True)
        else:
            st.write("No images found in the directory.")

if __name__ == '__main__':
    main()
