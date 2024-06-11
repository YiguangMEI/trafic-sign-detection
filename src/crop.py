import pandas as pd
import shutil
import os
from PIL import Image


def crop_images_from_folder(image_folder, csv_folder, output_folder):
    # S’assurer que les documents traduits existent
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Parcourez chaque fichier dans le dossier images
    for image_filename in os.listdir(image_folder):
        # Obtenir le nom du fichier et l’extension
        name, ext = os.path.splitext(image_filename)

        # Construire le fichier CSV correspondant
        csv_filename = f"{name}.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

        # Construire le chemin complet du fichier image
        image_path = os.path.join(image_folder, image_filename)

        # Vérifiez si le fichier CSV existe
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            print(f"csv_non_exists or empty：{csv_filename}")
            save_name = f"{name}_cropped_none_0.jpg"
            img = Image.open(image_path)
            img.save(os.path.join(output_folder, save_name))
            continue

        with open(csv_path, "r") as f:
            if f.read().isspace():
                # cropped_image_name = f"{name}_cropped_none_none.jpg"
                # cropped_image_path = os.path.join(output_folder, cropped_image_name)
                # shutil.copyfile(image_path,cropped_image_path)
                print(f"csv file contains only whitespace: {csv_filename}")
                save_name = f"{name}_cropped_none_0.jpg"
                img = Image.open(image_path)
                img.save(os.path.join(output_folder, save_name))
                continue

        # Lire un fichier CSV
        df = pd.read_csv(csv_path, header=None)

        # Ouvrir une image
        with Image.open(image_path) as img:
            # Parcourez chaque ligne du CSV
            for index, row in df.iterrows():
                xmin, ymin, xmax, ymax, label = row[0], row[1], row[2], row[3], row[4]

                # Recadrer une image
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                if label != "ff":
                    # Construire le chemin de sauvegarde de l’image recadrée avec le nom de la classification
                    cropped_image_name = f"{name}_cropped_{label}_{index}{ext}"
                    cropped_image_path = os.path.join(output_folder, cropped_image_name)

                    # Sauvegardez l’image recadrée

                    cropped_img.save(cropped_image_path)
                    # print(f"img：{cropped_image_path}")
