# Vision Transformer Replication & "Food Vision Big" Deployment

This project showcases a comprehensive journey through advanced deep learning concepts, from deeply understanding and replicating a seminal research paper to building and deploying a large-scale, real-world application.

## 🚀 Live Demo: Food Vision Big

Experience the deployed application live on Hugging Face Spaces! This app can classify an image into 101 different food categories.

**[➡️ Live Demo: Food Vision Big](https://huggingface.co/spaces/prithviraj-maurya/food-vision-big)**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/prithviraj-maurya/food-vision-big)

![Food Vision Big Demo](food_vision_big_demo.gif)

---

## 📖 Part 1: Replicating the Vision Transformer (ViT) Paper

The first part of this project involved a deep dive into the groundbreaking paper, **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**. The primary goal was not just to use the Vision Transformer (ViT) architecture, but to understand it from first principles.

### Key Activities:
- **Deconstructing the Paper:** Thoroughly analyzed the paper to understand the core mechanics, equations, and methodologies behind the ViT architecture.
- **Implementation from Scratch:** Replicated the entire ViT architecture, including:
  - **Image Patching:** Converting images into a sequence of flattened patches.
  - **Patch and Position Embeddings:** Creating learnable embeddings for image patches and their positions.
  - **Transformer Encoder:** Building the encoder block with its Multi-Head Self-Attention (MSA) and MLP layers.
- **Training and Validation:** Trained the custom-built ViT model on the "Pizza, Steak, Sushi" dataset to validate the implementation and replicate the paper's fundamental results.

This process is documented in detail within the `08_pytorch_paper_replicating.ipynb` notebook.

---

## 🍔 Part 2: "Food Vision Big" - A Deployed Image Classifier

The second part of the project focused on a more practical, application-oriented goal: building and deploying a robust food image classification system.

### Application Overview
"Food Vision Big" is a web application that leverages a state-of-the-art computer vision model to classify images of food into **101 different categories**.

### Model & Training
- **Model Architecture:** The application is powered by **EfficientNetV2-S**, a powerful and efficient model, utilized through transfer learning to achieve high accuracy.
- **Dataset:** The model was trained on the extensive **Food101 dataset**, which contains 101,000 images across 101 food classes.

### 💻 Local Training & Hardware
The `EfficientNetV2-S` model was trained locally on a powerful consumer-grade GPU, which was crucial for handling the large dataset and complex architecture.
- **GPU:** **NVIDIA GeForce RTX 4060**

### 🌐 UI and Deployment
- **Interactive Interface:** The user-friendly web interface was built using **Gradio**, allowing users to easily upload an image and receive a classification.
- **Deployment:** The final application is deployed and hosted on **Hugging Face Spaces**, making it publicly accessible to a global audience.

The entire journey of building, training, and deploying this application is detailed in the `09_pytorch_model_deployment.ipynb` notebook.

---

## 📂 Project Structure

- **`08_pytorch_paper_replicating.ipynb`**: A Jupyter notebook containing the complete code and theoretical explanations for replicating the Vision Transformer paper.
- **`09_pytorch_model_deployment.ipynb`**: A Jupyter notebook that details the process of training the `EfficientNetV2-S` model and building/deploying the Gradio application.
- **`requirements.txt`**: A file listing the Python dependencies required to run this project.

## ⚙️ How to Run

To explore this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebooks:**
    Launch Jupyter Notebook or JupyterLab and open the `.ipynb` files to see the code, explanations, and results.
    ```bash
    jupyter lab
    ```

## 🙏 Acknowledgements

The notebooks in this project are based on the fantastic curriculum from the [Zero to Mastery PyTorch for Deep Learning course](https://github.com/mrdbourke/pytorch-deep-learning) by Daniel Bourke.
