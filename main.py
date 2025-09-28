# main.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image

# -----------------------
# Define the same CNN as in training
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1,1,28,28)
            dummy = torch.relu(self.conv1(dummy))
            dummy = torch.relu(self.conv2(dummy))
            flattened_size = dummy.numel()  # 64*24*24 = 36864

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Load model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("mnist_cnn.pt", map_location=device)
model.eval()

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="MNIST Digit Classifier API")

# Transform function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("L")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1).item()
        return JSONResponse({"prediction": pred})
    except Exception as e:
        return JSONResponse({"error": str(e)})
