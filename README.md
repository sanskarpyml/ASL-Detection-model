<h1 align="center">🤟 ASL Alphabet Detection using CNN 🤖</h1>

<p align="center">
  <strong>A deep learning-based web app to recognize American Sign Language (A-Z) using image classification.</strong>
</p>

<p align="center">
  <img src="images/banner.png" alt="ASL Alphabet Classifier Banner" width="100%">
</p>

<h2>📌 Project Description</h2>
<p>
  This project is part of my internship at <strong>Unified Mentor</strong>, where I built an American Sign Language (ASL) alphabet classifier using Convolutional Neural Networks (CNNs). The model predicts 29 classes including A–Z and special signs like <em>Space</em>, <em>Delete</em>, and <em>Nothing</em>. It was trained using a curated dataset of <strong>1000 images per class</strong> and deployed using <strong>Streamlit</strong>.
</p>

<h2>📁 Dataset</h2>
<p>
  The dataset originally contained 3000 images per class, but for efficiency, I used 1000 images per class for this project. It contains images of hand gestures in RGB format, each representing one of the 29 labels.
</p>

<p align="center">
  <img src="images/dataset_grid.png" alt="ASL Dataset Overview" width="70%">
</p>

<ul>
  <li><strong>Classes:</strong> A-Z, SPACE, DELETE, NOTHING</li>
  <li><strong>Total Images Used:</strong> 29,000</li>
  <li><strong>Image Size:</strong> 200x200</li>
</ul>

<h2>🧠 Model Architecture</h2>
<p>
  The model uses a simple but powerful CNN architecture with data augmentation. It was trained using <code>tensorflow.keras</code> and achieved over <strong>95% accuracy</strong> on the validation set.
</p>

<p align="center">
  <img src="images/model_architecture.png" alt="Model Architecture" width="80%">
</p>

<h2>🔄 Workflow</h2>
<p>
  The overall pipeline of the ASL classifier is:
</p>
<ol>
  <li>Data Collection & Preprocessing</li>
  <li>Data Augmentation</li>
  <li>Model Building with CNN</li>
  <li>Training & Evaluation</li>
  <li>Deployment using Streamlit</li>
</ol>

<p align="center">
  <img src="images/workflow_diagram.png" alt="Workflow Diagram" width="80%">
</p>

<h2>📊 Model Evaluation</h2>
<p>
  The model was evaluated using accuracy, loss metrics, and a confusion matrix.
</p>

<p align="center">
  <img src="images/evaluation_chart.png" alt="Evaluation Charts" width="70%">
</p>

<ul>
  <li><strong>Training Accuracy:</strong> 96.8%</li>
  <li><strong>Validation Accuracy:</strong> 95.2%</li>
  <li><strong>Loss:</strong> CrossEntropy Loss</li>
</ul>

<h2>🚀 Deployment</h2>
<p>
  The model was deployed using <strong>Streamlit</strong>. The user can upload an image and get real-time predictions with confidence scores. Here's a look at the Streamlit app:
</p>

<p align="center">
  <img src="images/streamlit_demo.png" alt="Streamlit App Demo" width="75%">
</p>

<h2>🧪 Prediction Function</h2>
<p>
  The web app contains two functions:
</p>
<ul>
  <li><strong>load_and_preprocess_image()</strong> – to resize and scale new input images</li>
  <li><strong>predict_image_class()</strong> – to use the trained model for prediction</li>
</ul>


<h2>🛠️ Technologies Used</h2>
<ul>
  <li><strong>Python</strong></li>
  <li><strong>TensorFlow / Keras</strong></li>
  <li><strong>Streamlit</strong></li>
  <li><strong>Google Colab</strong></li>
  <li><strong>Jupyter Notebook</strong></li>
</ul>

<h2>🙋‍♂️ Author</h2>
<p>
  Created by <strong>Sanskar Gupta</strong> <br>
  Intern at <strong>Unified Mentor</strong> <br>
</p>


