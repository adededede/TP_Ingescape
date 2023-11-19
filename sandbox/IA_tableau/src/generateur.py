import base64
from io import BytesIO
import cv2
from tensorflow.keras.models import load_model

#CAS OU ON GENERE LES IMAGES PEU IMPORTE LA COULEUR
from imageGenerator.dcgan import DCGAN
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from PIL import Image

image_url_pink = ["https://render.fineartamerica.com/images/rendered/square-dynamic/small/images/artworkimages/mediumlarge/1/pink-roses-still-life-painting-2-nancy-merkle.jpg", "data:image/jpeg;base64,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", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRiIjudQ0prWtkUN0Zqxo_qJCQY3Ji3UwGWj2g1yKGC1opKF2KwStowQInhID6i1pOY4Rg", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSyA_hHsQxwAg6QqkIO6G2E4eajX5uXKEBGHw", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxWdRmTL5jjhxcuYHS56slRFQS65ceiARlaA", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTzG5E86ebQ9nzd6ml-5nwn7vl6BRvwCvyLrA", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSG50zqUjQ7pa6VCKRQ6ladfZrTyJSGia0xdg", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSBPvhg7BvjcXmcm9T-lncdUVZ_k3e8deCXgOydxTmLlQIXnrUrGYa-v_ztRNAsJ4EN0c8", "https://oic.uqam.ca/_next/image?url=https%3A%2F%2Fadmin.oic.production.nt2.ca%2Fwp-content%2Fuploads%2F2022%2F03%2Fcouv_9-e1684334002690.jpg&w=256&q=75", "https://e.snmc.io/i/1200/s/cd4ab59f14995103a800c7d3582291b0/1580355"]
image_url_red = ["https://developers.google.com/earth-engine/datasets/images/NOAA/NOAA_CDR_PATMOSX_V53_sample.png", "https://hexoa.fr/9938-product_miniature/affiche-abstrait-damiers.webp", "https://i.pinimg.com/474x/96/bd/9e/96bd9ef03bf75b2d89dbcfb2027f08c8.jpg", "https://styles.redditmedia.com/t5_35ra27/styles/profileIcon_h8h771hdtsya1.jpg?width=256&height=256&frame=1&auto=webp&crop=256:256,smart&s=9ff822513967dd543a02ad006c8d6b478361e782"]
image_url_green = ["https://i.pinimg.com/474x/ed/d7/8e/edd78e55a7666faa52e54b538d64e2f3.jpg", "https://i.pinimg.com/474x/5c/6b/df/5c6bdf07568505d730fe711a1dc199df.jpg", "https://i.pinimg.com/474x/6d/d0/f3/6dd0f360f3e6d53f6a46d77de3a78655.jpg", "https://i.pinimg.com/474x/3a/0d/eb/3a0debcb4095146bda67e3eb4ed1d0ee.jpg", "https://d7hftxdivxxvm.cloudfront.net/?height=256&quality=80&resize_to=fill&src=https%3A%2F%2Fd32dm0rphc51dk.cloudfront.net%2FyvSvFJZ9Xsd8mFA5LMEMEw%2Flarger.jpg&width=256", "https://d7hftxdivxxvm.cloudfront.net/?height=256&quality=80&resize_to=fill&src=https%3A%2F%2Fd32dm0rphc51dk.cloudfront.net%2F94nHNvNP9xSJ01dASy1l1w%2Flarger.jpg&width=256"]
image_url_yellow = ["https://i.pinimg.com/474x/1f/09/45/1f09453b88f83b2af9ecedbea14f7de0.jpg", "https://storage.googleapis.com/dream-machines-output/174af5c1-dd8e-4015-b418-6bdd434affbf/0_1.png", "https://i.pinimg.com/474x/1b/10/64/1b1064dd282b356c5b57063e6d70b58e.jpg"]
image_url_blue = ["https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSG50zqUjQ7pa6VCKRQ6ladfZrTyJSGia0xdg", "https://i.pinimg.com/474x/62/fa/8d/62fa8de180ffd7c471df0615e365dd22.jpg", "https://www.meisterdrucke.us/image_getborder_hard_V05.php?thumb=1&id=981&iid=1318764&form_width=60&form_height=60", "https://d7hftxdivxxvm.cloudfront.net/?height=256&quality=80&resize_to=fill&src=https%3A%2F%2Fd32dm0rphc51dk.cloudfront.net%2FMHoTVGWB7nI_pRG8AgPpag%2Flarger.jpg&width=256", "https://news.artnet.com/app/news-upload/2017/03/blueboy_872-693x1024-1-256x256.png"]
image_url = [image_url_pink,image_url_red,image_url_yellow,image_url_blue,image_url_green]

def generation_image_v2(couleur):
    if len(couleur)<4:
        # pas de couleur spécifié
        liste_couleur = random.randint(0, len(image_url))
        index = random.randint(0, len(image_url[liste_couleur])-1)
        return (image_url[liste_couleur])[index]
    if (re.compile(r'(rouge)')).search(couleur):
        index = random.randint(0, len(image_url_red)-1)
        return image_url_red[index]
    elif (re.compile(r'(rose)')).search(couleur):
        index = random.randint(0, len(image_url_pink)-1)
        return image_url_pink[index]
    elif (re.compile(r'(jaune)')).search(couleur):
        index = random.randint(0, len(image_url_yellow)-1)
        return image_url_yellow[index]
    elif (re.compile(r'(vert)')).search(couleur):
        index = random.randint(0, len(image_url_green)-1)
        return image_url_green[index]
    elif (re.compile(r'(bleu)')).search(couleur):
        index = random.randint(0, len(image_url_blue)-1)
        return image_url_blue[index]
    else:
        # la couleur demandé n'est pas dispo
        return 0


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

def sauver_image(image_array):
    # Check the data type and convert if needed
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)

    # Check the shape and adjust if needed
    if image_array.shape[0] == 1 and image_array.shape[1] == 1:
        # If the shape is (1, 1, 3), it might be a single pixel, adjust as needed
        # For example, you can repeat the pixel to create a small image
        image_array = np.repeat(np.repeat(image_array, 10, axis=0), 10, axis=1)

    # Save the image to a specific destination using OpenCV
    chemin = './image_courante/image.png'
    retour = cv2.imwrite(chemin, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return retour

if __name__ == "__main__":
    image_1 = generation_image("rose")
    # image_2 = generation_image_sans_couleur()
    plt.imshow(image_1)
    plt.axis('off')
    plt.show()
    sauver_image(image_1)
    # plt.imshow(image_2)
    # plt.axis('off')
    # plt.show()