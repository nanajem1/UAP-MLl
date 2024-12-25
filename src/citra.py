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

    organic_classes = ["coffee_grounds", "eggshells", "food_waste", "tea_bags"]
    category = "Organik" if class_name in organic_classes else "Non-Organik"

    return class_name, category, round(confidence, 2)  # Batasi confidence menjadi 2 angka
