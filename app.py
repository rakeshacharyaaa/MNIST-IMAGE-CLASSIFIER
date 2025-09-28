# app.py
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Load the trained MNIST model
model = torch.jit.load("mnist_cnn.pt", map_location="cpu")
model.eval()

# Transform function for input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(img):
    """
    Takes a PIL image (or numpy array), transforms it, and predicts the digit
    """
    if isinstance(img, Image.Image):
        image = img
    else:
        image = Image.fromarray(img).convert("L")
    
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return str(pred)

# Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)"),
    outputs=gr.Label(label="Predicted Digit"),
    title="MNIST Digit Classifier",
    description="Draw a digit and the model will predict what it is!"
)

if __name__ == "__main__":
    demo.launch()
