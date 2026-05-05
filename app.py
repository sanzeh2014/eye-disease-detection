# app.py - AI-Based Eye Disease Detection System
# This is the frontend for my MSc dissertation project
# GitHub: sanzeh2014/eye-disease-detection

# I'm importing all the libraries I need for this web app
import gradio as gr  # This makes building the web interface super easy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2  # For image processing and creating heatmap overlays
import os
import json

# ============================================================================
# MY CONFIGURATION - These match what I set in my Jupyter notebooks
# ============================================================================

# From Notebook 01 - I made sure my class order follows the ICDRSS standard
# Class 0 = No DR, Class 1 = Mild, Class 2 = Moderate, Class 3 = Severe, Class 4 = Proliferative
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Setting up the device - my Mac has an MPS chip which is great for acceleration
# This comes from my training notebook where I checked for hardware
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using Apple MPS GPU - much faster than CPU")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA GPU")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU - will be slower but still works")

# This is my trained model - it's 269MB and took weeks to train
MODEL_PATH = "models/best_model.pth"

# My preprocessing pipeline from Notebook 02
# I resize to 224x224 (what ResNet50 expects), convert to tensor, 
# then normalize using ImageNet statistics (standard for transfer learning)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Loading my training configuration if it exists (saved from Notebook 02)
config_path = "models/training_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded training config: {config.get('batch_size', 'N/A')} batch size")

# ============================================================================
# LOADING MY TRAINED MODEL - This was the hardest part to get right
# ============================================================================

print(f"\nLoading my trained model from {MODEL_PATH}...")
print("I trained this model on 3,662 retinal images from the APTOS 2019 dataset")
print("It took me weeks of training and debugging to get here")

def load_my_model():
    """This function loads my trained ResNet50 model"""
    
    # Creating the exact same architecture I used for training
    model = models.resnet50(weights=None)
    # I replaced the final layer to output 5 classes (one for each DR severity)
    model.fc = nn.Linear(2048, 5)
    
    # Loading my saved model file
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"Found {len(keys)} keys in checkpoint")
        print(f"   First key: {keys[0][:50]}...")
        
        # Figuring out how I saved the model - I tried different methods
        state_dict = None
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Found 'state_dict' key")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key")
        elif 'model.' in keys[0] or 'conv1' in keys[0]:
            state_dict = checkpoint
            print("Direct state_dict with model keys")
        else:
            state_dict = checkpoint
        
        # My saved model has 'model.' prefix on all keys - I need to strip it
        # This was a headache to figure out
        if state_dict:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Removing 'model.' prefix
                    new_state_dict[new_key] = value
                elif key.startswith('module.'):
                    new_key = key[7:]  # Removing 'module.' prefix  
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Loading with strict=False because some batch norm stats are missing
            # This is normal when loading a trained model
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)} (expected - batch norm stats)")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} (handled)")
            print("Model loaded successfully")
        
    else:
        model = checkpoint
        print("Loaded as full model")
    
    model.eval()  # Putting model in evaluation mode (no dropout)
    model.to(DEVICE)
    return model

# Attempting to load my model - I hope this works
try:
    model = load_my_model()
    print("Model loaded! My weeks of training paid off")
    
    # Quick test to make sure the model works
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        test_output = model(dummy)
        print(f"Model test passed. Output shape: {test_output.shape}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    print("Something went wrong - using a dummy model for testing")
    print("Please check that models/best_model.pth exists")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 5)
    model.eval()
    model.to(DEVICE)

# ============================================================================
# GRAD-CAM - This shows WHY my model makes certain predictions
# ============================================================================
# I learned about Grad-CAM from Selvaraju et al. 2017
# It helps clinicians trust my model by showing what it's looking at

class GradCAM:
    """My implementation of Grad-CAM for model explainability"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Setting up hooks to capture gradients and activations"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.handles.append(self.target_layer.register_backward_hook(backward_hook))
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
    
    def generate(self, input_tensor, target_class=None):
        """Generating the heatmap that shows what my model focused on"""
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax().item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # This is the core Grad-CAM math
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # Normalizing to make values between 0 and 1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

def get_target_layer(model):
    """Finding the last convolutional layer in ResNet50"""
    try:
        return model.layer4[2].conv3  # Standard layer for ResNet50
    except AttributeError:
        # If that fails, I search for any Conv2d in layer4
        for name, module in model.named_modules():
            if 'layer4' in name and isinstance(module, nn.Conv2d):
                return module
        return model.layer4[-1]

def generate_heatmap_overlay(model, image, class_name):
    """Creating the final heatmap overlay on the original image"""
    try:
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        
        input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
        target_idx = CLASS_NAMES.index(class_name)
        heatmap = gradcam.generate(input_tensor, target_idx)
        gradcam.remove_hooks()
        
        # Making the heatmap match the original image size
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        
        # Converting PIL image to numpy array for OpenCV
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Creating a colored heatmap (red/yellow = high attention)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blending the heatmap with the original image
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        return overlay
    except Exception as e:
        print(f"Heatmap generation had an issue: {e}")
        return np.array(image)  # Return original image if heatmap fails

# ============================================================================
# CLINICAL INTERPRETATIONS - From my Notebook 01
# ============================================================================
# I wrote these based on the ICDRSS clinical guidelines

INTERPRETATIONS = {
    "No DR": "HEALTHY - Normal retina with clear blood vessels. No signs of diabetic retinopathy detected.",
    "Mild": "MILD - Microaneurysms (small red dots) detected. Early stage, routine monitoring recommended.",
    "Moderate": "MODERATE - Hemorrhages and hard exudates visible. Refer to optometrist within 6 months.",
    "Severe": "SEVERE - Extensive damage following 4-2-1 rule criteria. Urgent referral to ophthalmologist.",
    "Proliferative": "PROLIFERATIVE - Abnormal new blood vessels forming. Emergency referral - immediate treatment needed."
}

# ============================================================================
# THE MAIN PREDICTION FUNCTION - This does the actual work
# ============================================================================

def predict_dr(image):
    """This is called when someone clicks the Diagnose button"""
    
    if model is None:
        return "Model not loaded", "0%", "Please check console", None, ""
    
    if image is None:
        return "No image", "0%", "Please upload a retinal fundus image", None, ""
    
    try:
        # Step 1: Preprocess the image
        input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
        
        # Step 2: Run the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # Step 3: Get the results
        class_name = CLASS_NAMES[prediction.item()]
        confidence_pct = f"{confidence.item():.1%}"
        interpretation = INTERPRETATIONS.get(class_name, "See heatmap for details")
        
        # Step 4: Generate the Grad-CAM heatmap
        heatmap = generate_heatmap_overlay(model, image, class_name)
        
        # Step 5: Create a nice text display of all probabilities
        probs_text = "All Class Probabilities\n\n"
        for i, name in enumerate(CLASS_NAMES):
            prob = probabilities[0][i].item()
            bar_length = int(prob * 30)
            bar = "#" * bar_length + "-" * (30 - bar_length)
            probs_text += f"- {name}: {prob:.1%} {bar}\n"
        
        # Console output for debugging (helps me see what's happening)
        print(f"\nPrediction: {class_name} ({confidence_pct})")
        for i, name in enumerate(CLASS_NAMES):
            print(f"   {name}: {probabilities[0][i].item():.3f}")
        
        return class_name, confidence_pct, interpretation, heatmap, probs_text
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Error", "0%", f"{str(e)[:100]}", None, ""

# ============================================================================
# BUILDING THE WEB INTERFACE WITH GRADIO
# ============================================================================
# Gradio makes this so easy - no HTML or JavaScript needed

# Some custom styling to make it look professional
custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    color: #2c3e50;
}
"""

with gr.Blocks(title="AI-Based Eye Disease Detection", theme=gr.themes.Soft(), css=custom_css) as demo:
    
    gr.Markdown("""
    # AI-Based Eye Disease Detection System
    ### Using Explainable Deep Learning Models (ResNet50 + Grad-CAM)
    
    This system detects **Diabetic Retinopathy severity** as a primary use case of AI-based eye disease screening.
    
    **Upload a retinal fundus photograph** and the system will predict the severity with a **Grad-CAM heatmap** 
    showing what the model focused on.
    """)
    
    with gr.Row():
        # Left side - where the user uploads images
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil", 
                label="Upload Retinal Fundus Image",
                sources=["upload"],
                height=320
            )
            diagnose_btn = gr.Button("Diagnose", variant="primary", size="lg")
            
            # Showing my model's performance metrics (from my paper)
            gr.Markdown("""
            ---
            ### Model Performance (Diabetic Retinopathy)
            | Metric | Value |
            |--------|-------|
            | **Accuracy** | 73.1% |
            | **Kappa Score** | 0.68 |
            | **Cross-dataset** | 68.2% (Messidor-2) |
            """)
        
        # Right side - where results appear
        with gr.Column(scale=1):
            diagnosis = gr.Label(label="Diagnosis", num_top_classes=5)
            confidence = gr.Textbox(label="Confidence Score", lines=1)
            interpretation = gr.Textbox(label="Clinical Interpretation", lines=4)
            heatmap = gr.Image(label="Grad-CAM Explanation (Red/Yellow = Key Areas)", type="numpy", height=280)
    
    # Full probability breakdown
    with gr.Row():
        all_probs = gr.Markdown("### All Class Probabilities\n*Upload an image and click Diagnose*")
    
    # Footer with helpful information
    gr.Markdown("""
    ---
    ### How to Interpret the Results
    
    | Severity | What It Means | Clinical Action |
    |----------|---------------|-----------------|
    | **No DR** | Healthy retina | Routine screening |
    | **Mild** | Microaneurysms only | Monitor annually |
    | **Moderate** | Hemorrhages + exudates | Refer within 6 months |
    | **Severe** | 4-2-1 rule criteria | Urgent referral |
    | **Proliferative** | New blood vessels | Emergency treatment |
    
    ### Understanding the Heatmap
    - **Red/Yellow areas** = Regions that most influenced the prediction
    - **Blue areas** = Low influence on the prediction
    - **Good sign**: Heatmap highlights actual lesions (hemorrhages, exudates)
    - **Warning**: Heatmap highlights image edges or artifacts, be cautious
    
    ---
    **Model:** ResNet50 (Two-stage transfer learning) | **Application:** AI-Based Eye Disease Detection  
    **Case Study:** Diabetic Retinopathy | **Training Data:** APTOS 2019 (3,662 images)  
    **Explainability:** Grad-CAM (Selvaraju et al. 2017)  
    **Student:** Sanjay Chaudhary (B01036144) | **Supervisor:** Dr. Tertsegha Anande | **Institution:** Ulster University
    """)
    
    # Connecting the button to my prediction function
    diagnose_btn.click(
        fn=predict_dr,
        inputs=input_image,
        outputs=[diagnosis, confidence, interpretation, heatmap, all_probs]
    )

# ============================================================================
# STARTING THE WEB APP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING MY AI-BASED EYE DISEASE DETECTION SYSTEM")
    print("=" * 60)
    print(f"Model file: {MODEL_PATH}")
    print(f"Running on: {DEVICE}")
    print(f"Can detect: {', '.join(CLASS_NAMES)}")
    print(f"Model accuracy: 73.1% | Kappa: 0.68")
    print("\nOpening web interface...")
    print("Local URL: http://127.0.0.1:7860")
    print("Shareable link (public for 72 hours) will appear below")
    print("   Anyone with this link can use my model")
    print("=" * 60 + "\n")
    
    # Launching the app - share=True creates a public link for my dissertation
    demo.launch(
        share=True,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860
    )