#Machine Learning & Deep Learning Architectures

  This project integrates three distinct mini-projects, each covering a specific application of machine learning and deep learning. The aim is to provide insights into CNN architectures for image classification, develop an interactive NLP tool, and create a Seq2Seq model with an attention mechanism for sequence transformation.

#Project Structure:

 *CNN Architecture Comparison
 *NLP Interactive Tool
 *Seq2Seq Model with Attention Mechanism
 
** Note: For these tasks or projects, you’ll need high memory capacity and a stable, high-speed internet connection. For example, if you're working on Google Colab, connect the runtime to a T4 GPU to optimize performance.**

#1. CNN Architecture Comparison:

*Overview:

  This project compares the performance of different Convolutional Neural Network (CNN) architectures on three popular image datasets: MNIST, Fashion MNIST (FMNIST), and CIFAR-10. The goal is to analyze which CNN architecture performs best on each dataset by tracking loss curves and accuracy metrics.

*Datasets:

    MNIST   : Handwritten digits (0–9).
    FMNIST  : Fashion items categorized into 10 classes.
    CIFAR-10: 60,000 32x32 color images across 10 classes.

*Architectures:

    Various CNN architectures are explored,LeNet, AlexNet, ResNet,VGG, GoogleNet, Xception and ResNet.

*Libraries Used:

    PyTorch    : For deep learning and building CNN models.
    Matplotlib : To plot and visualize loss and accuracy curves.

*Steps: For this task, refer to cnn_project.py file.

1. Load and preprocess each dataset.
2. Define and initialize different CNN architectures.
3. Train each architecture on the datasets, tracking the loss and accuracy over epochs.
4. Plot loss curves and accuracy metrics for analysis.
5. Identify the best-performing architecture for each dataset.

*Evaluation Metrics:

    Loss Curves: Track model learning progression.
    Accuracy: Evaluate the overall classification performance on each dataset.


#2. NLP Interactive Tool:

*Overview:

  This project implements an interactive NLP tool using Streamlit as the front end. Users can select specific NLP tasks, input prompts, and receive outputs in real time. Each task utilizes pre-trained models from Hugging Face’s model library for backend processing.

*Features:

    Next Word Prediction : Predicts possible continuations for a prompt.
    Text Generation      : Generates coherent text from a given prompt.
    Chatbot              : Engages in conversational AI.
    Sentiment Analysis   : Determines the emotional tone of text.
    Question Answering   : Provides responses based on a given context.
    Text Summarization   : Summarizes lengthy text passages.
    Image Generation     : Creates images from text descriptions.

*Libraries Used:

    Streamlit                 : To create the user-friendly web interface.
    Hugging Face Transformers : To access pre-trained models for various NLP tasks.
    PyTorch					  : For model inference.
    Diffusers and PIL		  : For image generation and processing.

*Steps: For this task, refer to nlp_project.py file.

1.Load pre-trained models for each NLP task from Hugging Face.
2.Define task-specific functions to process user inputs and generate outputs.
3.Set up Streamlit for the user interface.
4.Display results interactively in Streamlit.
5.Implement user feedback collection to evaluate tool performance.


#3. Seq2Seq Model with Attention Mechanism:

*Overview:

  This project focuses on a Seq2Seq model with an attention mechanism, trained on synthetic data. The source sequence consists of a random sequence of integers, while the target sequence is the reverse of the source. The model learns this transformation using attention to improve performance.

*Synthetic Dataset:

    Source Sequence : A random sequence of integers.
    Target Sequence : The reversed source sequence.

*Model Architecture:

    A Seq2Seq Encoder-Decoder structure with an attention mechanism to focus on relevant parts of the sequence.

*Libraries Used:

    PyTorch    : For model construction, training, and defining attention mechanisms.
    Numpy      : For creating synthetic sequences.
    Matplotlib : To visualize the loss curve.

*Steps: For this task, refer to seq2seq_project.py file.

1.Generate a synthetic dataset where the target sequence is the reverse of the source.
2.Define the Encoder, Decoder, and Attention modules.
3.Implement a Seq2Seq model that incorporates the attention mechanism.
4.Train the model on the synthetic data, tracking loss across epochs.
5.Evaluate the model using accuracy, precision, recall, and F1-score.

*Evaluation Metrics:

    Loss Curve : Monitor learning progression at each epoch.
    Accuracy, Precision, Recall, and F1-score : Measure model performance on sequence transformation tasks.
