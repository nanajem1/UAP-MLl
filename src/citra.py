import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/mobilenet_model.h5"
model = load_model(MODEL_PATH)

class_names = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "clothing", "coffee_grounds", "disposable_plastic_cutlery",
    "eggshells", "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups",
    "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
    "plastic_trash_bags", "plastic_water_bottles", "shoes", "steel_food_cans",
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

def predict(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Tambahkan ambang batas keyakinan
    THRESHOLD = 0.5  # 50% confidence threshold
    if confidence < THRESHOLD:
        return "Unknown", "Bukan Sampah", round(confidence, 2)  # Batasi confidence menjadi 2 angka

    class_name = class_names[class_index]

    # Klasifikasi kategori sampah
    organic_classes = ["coffee_grounds", "eggshells", "food_waste", "tea_bags"]
    plastic_classes = [
        "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
        "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
        "plastic_trash_bags", "plastic_water_bottles", "disposable_plastic_cutlery"
    ]
    paper_classes = ["newspaper", "office_paper", "magazines", "paper_cups"]
    cardboard_classes = ["cardboard_boxes", "cardboard_packaging"]
    glass_classes = ["glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars"]
    metal_classes = ["aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "steel_food_cans"]
    textile_classes = ["clothing", "shoes"]

    # Tentukan kategori
    if class_name in organic_classes:
        category = "Organik"
    elif class_name in plastic_classes:
        category = "Non-Organik - Plastik"
    elif class_name in paper_classes:
        category = "Non-Organik - Kertas"
    elif class_name in cardboard_classes:
        category = "Non-Organik - Karton"
    elif class_name in glass_classes:
        category = "Non-Organik - Kaca"
    elif class_name in metal_classes:
        category = "Non-Organik - Logam"
    elif class_name in textile_classes:
        category = "Non-Organik - Tekstil"
    else:
        category = "Non-Organik - Lainnya"

    return class_name, category, round(confidence, 2)  # Batasi confidence menjadi 2 angka
