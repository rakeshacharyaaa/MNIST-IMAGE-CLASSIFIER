# MNIST Image Classifier API

A **FastAPI API** that predicts handwritten digits (0-9) using a **PyTorch Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  

Users can **upload an image of a handwritten digit** and get an **instant prediction**.

---

## Features

- Predict handwritten digits (0-9)  
- Built with **PyTorch** for the CNN model  
- Exposed via **FastAPI**  
- API documentation automatically generated with **Swagger UI**  
- Easy deployment on cloud platforms like **Render.com**  

---

## Project Structure

mnist-image-classifier/
│
├─ main.py # FastAPI API
├─ train_mnist.py # Script to train CNN model
├─ mnist_cnn.pt # Trained model (PyTorch)
├─ requirements.txt # Dependencies
├─ .gitignore # Ignore unnecessary files/folders
├─ README.md # Project documentation


---

## Installation (Local)

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/mnist-image-classifier.git
cd mnist-image-classifier


Create a virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate



Install dependencies

pip install -r requirements.txt

Running Locally

Start the FastAPI server:

uvicorn main:app --reload



Open your browser:

http://127.0.0.1:8000/docs

Use the Swagger UI to upload a handwritten digit image and get predictions.



Deployment

This project can be deployed to free cloud platforms such as Render.com:

Push your repository to GitHub.

Connect the GitHub repo to Render.com.

Set Start Command:

uvicorn main:app --host 0.0.0.0 --port 10000

Deploy → get a public URL for your live API.

Usage Example (Swagger UI)

Go to /docs in your browser

Click POST /predict/

Upload an image of a handwritten digit (28x28 or any size)

Click Execute → see predicted digit in response



Team Members

NOEL QUADRAS - 01SU22CS118

PRINSTON DSOUZA - 01SU22CS138

RAKESH - 01SU22CS145

SUMAN PRABHU - 01SU22CS208



Notes

The mnist_cnn.pt file is the trained PyTorch model.

The MNIST dataset downloads automatically when running train_mnist.py.

For deployment, venv/ and __pycache__/ are ignored via .gitignore.



License

This project is for educational purposes.


---



