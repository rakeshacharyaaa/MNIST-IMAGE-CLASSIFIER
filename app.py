# app.py
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
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
    Attempts both original and inverted prediction.
    """
    if isinstance(img, Image.Image):
        image = img
    else:
        image = Image.fromarray(img).convert("L")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    # If likely confusion (only 0 or 8), also try inverted
    if pred == 0 or pred == 8:
        inverted = ImageOps.invert(image.convert("L"))
        tensor_inv = transform(inverted).unsqueeze(0)
        with torch.no_grad():
            output_inv = model(tensor_inv)
            pred_inv = output_inv.argmax(dim=1).item()
        # Simple heuristic: return the prediction further from 0 if orig is 0 but inv isn't
        if pred == 0 and pred_inv != 0:
            return str(pred_inv)
        # If both are same, just return pred
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
