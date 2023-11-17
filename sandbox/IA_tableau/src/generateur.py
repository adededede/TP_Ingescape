import base64
from io import BytesIO
from tensorflow.keras.models import load_model

#CAS OU ON GENERE LES IMAGES PEU IMPORTE LA COULEUR
from imageGenerator.dcgan import DCGAN
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from PIL import Image



def generation_image_sans_couleur():
    dcgan = DCGAN(img_rows=128, img_cols=128, channels=3, latent_dim=256)
    dcgan.load_weights(generator_file='d:/Ingescape/sandbox/IA_tableau/src/imageGenerator/generator (fluid_256_128).h5')
    def generate_latent_points(latent_dim, n_samples):
        x_input = np.random.randn(latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    latent_points = generate_latent_points(256, 20)

    generated_images = dcgan.generator.predict(latent_points)

    return generated_images[random.randint(0,len(generated_images))]


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# Fonction pour calculer le pourcentage d'une couleur spécifique dans une image
def calculate_color_percentage(image, color):
    if color == "pink":
        return np.mean(np.all(image > [0.98, 0.75, 0.75], axis=-1))
    elif color == "blue":
        return np.mean(image[:,:,2] > 0.98)
    elif color == "red":
        return np.mean(np.all(image > [0.98, 0.1, 0.1], axis=-1))
    elif color == "yellow":
        return np.mean(np.all(image > [0.98, 0.9, 0.1], axis=-1))
    elif color == "green":
        return np.mean(image[:,:,1] > 0.98)
    elif color == "black":
        return np.mean(np.all(image < 0.2, axis=-1))
    else:
        return 0

def get_couleur(couleur):
    if (re.compile(r'(rouge)')).search(couleur):
        return "red"
    elif (re.compile(r'(rose)')).search(couleur):
        return "pink"
    elif (re.compile(r'(jaune)')).search(couleur):
        return "yellow"
    elif (re.compile(r'(noir)')).search(couleur):
        return "black"
    elif (re.compile(r'(vert)')).search(couleur):
        return "green"
    elif (re.compile(r'(bleu)')).search(couleur):
        return "blue"
    else:
        # la couleur demandé n'est pas dispo
        return 0

def generation_image(couleur):
    dcgan = DCGAN(img_rows=128, img_cols=128, channels=3, latent_dim=256)
    dcgan.load_weights(generator_file='d:/Ingescape/sandbox/IA_tableau/src/imageGenerator/generator (fluid_256_128).h5')

    latent_points = generate_latent_points(256, 20)
    generated_images = dcgan.generator.predict(latent_points)

    color = get_couleur(couleur)
    color_percentages = [calculate_color_percentage(img, color) for img in generated_images]

    # Trouver l'index de l'image avec le pourcentage le plus élevé de la couleur choisie
    max_color_index = np.argmax(color_percentages)
    image = generated_images[max_color_index]

    return image

# if __name__ == "__main__":
#     image_1 = generation_image("rose")
    # image_2 = generation_image_sans_couleur()
    # plt.imshow(image_1)
    # plt.axis('off')
    # plt.show()
    # plt.imshow(image_2)
    # plt.axis('off')
    # plt.show()