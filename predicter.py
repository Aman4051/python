import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp

# ===================================================================
# 1. CONSTANTS AND DATABASES
# ===================================================================

# These would be loaded from your meta.json files
PART_CLASSES = 16  # Total number of part classes + 1 for background
DAMAGE_CLASSES = 6 # Total number of damage classes + 1 for background

PART_ID_TO_NAME = {1: "bumper", 2: "door", 5: "headlight"}
DAMAGE_ID_TO_NAME = {1: "scratch", 2: "dent", 3: "cracked"}

COST_DATABASE = {
    "Toyota Camry": {
        "bumper": {"scratch": 150, "dent": 300, "cracked": 800, "replacement": 800},
        "door": {"scratch": 200, "dent": 500, "cracked": 1200, "replacement": 1200},
        "headlight": {"scratch": 50, "dent": 250, "cracked": 400, "replacement": 400}
    },
    "BMW X5": {
        "bumper": {"scratch": 400, "dent": 900, "cracked": 2000, "replacement": 2000},
        "door": {"scratch": 600, "dent": 1500, "cracked": 3000, "replacement": 3000},
        "headlight": {"scratch": 150, "dent": 700, "cracked": 1200, "replacement": 1200}
    }
}

# ===================================================================
# 2. HELPER FUNCTIONS
# ===================================================================

def prepare_image(image_pil):
    """Prepares a user-uploaded PIL image for the model."""
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)

def get_prediction_mask(model, image_tensor):
    """Gets a raw prediction from a model."""
    with torch.no_grad():
        output = model(image_tensor)
    return torch.argmax(output, dim=1).squeeze()

# ===================================================================
# 3. MAIN LOGIC FUNCTIONS
# ===================================================================

def load_models(part_model_path, damage_model_path):
    """Loads the two trained models."""
    part_model = smp.MAnet(encoder_name="resnet50", classes=PART_CLASSES)
    part_model.load_state_dict(torch.load(part_model_path, map_location="cpu"))
    part_model.eval()

    damage_model = smp.MAnet(encoder_name="resnet50", classes=DAMAGE_CLASSES)
    damage_model.load_state_dict(torch.load(damage_model_path, map_location="cpu"))
    damage_model.eval()
    
    return part_model, damage_model

def calculate_final_quote(car_model, parts_mask, damages_mask):
    """Calculates the final cost based on the predicted masks."""
    total_cost = 0
    cost_breakdown = []
    damage_analysis = []
    
    unique_part_ids = torch.unique(parts_mask).tolist()
    is_damaged_mask = (damages_mask > 0)

    for part_id in unique_part_ids:
        if part_id == 0: continue
        part_name = PART_ID_TO_NAME.get(part_id, "unknown_part")
        is_current_part_mask = (parts_mask == part_id)

        part_area = torch.sum(is_current_part_mask).item()
        if part_area == 0: continue
        
        damaged_area = torch.sum(is_current_part_mask & is_damaged_mask).item()
        damage_percentage = (damaged_area / part_area) * 100
        damage_analysis.append(f"- {part_name.title()}: {damage_percentage:.1f}% damaged")

        if damage_percentage > 50.0:
            cost = COST_DATABASE[car_model][part_name]["replacement"]
            cost_breakdown.append(f"- {part_name.title()} needs REPLACEMENT: ${cost}")
            total_cost += cost
        else:
            unique_damage_ids = torch.unique(damages_mask[is_current_part_mask]).tolist()
            for damage_id in unique_damage_ids:
                if damage_id == 0: continue
                damage_name = DAMAGE_ID_TO_NAME.get(damage_id, "unknown_damage")
                if damage_name in COST_DATABASE[car_model][part_name]:
                    cost = COST_DATABASE[car_model][part_name][damage_name]
                    cost_breakdown.append(f"- {part_name.title()} has {damage_name}: ${cost}")
                    total_cost += cost
    
    return f"${total_cost}", "\n".join(cost_breakdown), "\n".join(damage_analysis)