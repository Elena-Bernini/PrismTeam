# PrismPlan

### AI-Assisted Floor Plan Generation and Retrieval

![Status](https://img.shields.io/badge/status-in%20development-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)
![License](https://img.shields.io/badge/license-Academic-lightgrey)

An AI-powered system for floor plan retrieval and generation, developed by **Prism Team** in collaboration with **Maticad**.

---

## 👥 Team

**Prism Team**

* Elena Bernini
* Piergiorgio Signorino

---

## 📖 Project Overview

Designing interior spaces requires balancing structural constraints, functional requirements, and aesthetic considerations. While modern CAD software provides powerful design tools, the early conceptual phase remains largely manual and time-consuming.

**PrismPlan** explores how Artificial Intelligence can assist designers by automatically retrieving existing layouts and generating new floor plan proposals from partial user specifications.

The project combines Graph Neural Networks, similarity search techniques, and generative AI models to create an intelligent assistant capable of supporting architects and interior designers throughout the design process.

---

## 🎯 Objectives

The project focuses on two complementary tasks:

### Floor Plan Retrieval

Given a partial floor plan, the system retrieves the most similar completed layouts from a database of professional CAD designs.

Main components:

* Graph-based floor plan representation
* Graph Neural Network (GNN) embeddings
* Similarity search with FAISS
* Ranking and recommendation of layouts

### Floor Plan Generation

Given incomplete specifications, the system generates new floor plans while respecting architectural constraints.

Possible inputs include:

* Number of rooms
* Room categories
* Area requirements
* Adjacency constraints
* Partial floor plan sketches

Generated layouts should be both realistic and structurally valid.

---

## 🏗 Dataset

### Maticad Dataset

Professional interior design projects provided by Maticad.

### RPLAN Dataset

Public benchmark dataset used for training, experimentation, and evaluation.

The datasets include:

* Room types
* Room dimensions
* Connectivity information
* Spatial relationships
* Architectural constraints

---

## 🧠 Methodology

### Graph Representation

Each floor plan is represented as an attributed graph:

**Nodes**

* Room type
* Area
* Additional room features

**Edges**

* Adjacency relationships
* Doors and passages
* Connectivity information

### Retrieval Pipeline

1. Floor plan preprocessing
2. Graph construction
3. GNN embedding generation
4. Vector indexing with FAISS
5. Similarity-based retrieval

### Generation Pipeline

The project will investigate multiple approaches:

* Diffusion-based layout generation
* Transformer-based generation
* Constraint-aware decoding
* Interactive refinement mechanisms

### Human-in-the-Loop Refinement

Users can iteratively improve generated designs:

1. Generate candidate layouts
2. Provide corrections
3. Update constraints
4. Generate refined solutions

---

## 🛠 Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* FAISS
* Hugging Face Diffusers
* NumPy
* Pandas
* NetworkX
* Matplotlib

---

## 📂 Repository Structure


### Retrieval Metrics

* Top-k Accuracy
* Recall@k
* Mean Average Precision (mAP)

### Generation Metrics

* Constraint Satisfaction Rate
* Layout Validity
* Diversity Score
* User Preference Evaluation

---

## 📚 References

* Nauata, N. et al. *House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects*. CVPR 2021.

* Chai, W. et al. *LayoutDM: Transformer-based Diffusion Model for Layout Generation*. CVPR 2023.

* Liu, C. et al. *Data-driven Interior Plan Generation for Residential Buildings*. ACM SIGGRAPH Asia 2019.

---

## 🚀 Future Developments

* Interactive CAD integration
* Natural language floor plan generation
* Multi-floor building layouts
* Design optimization through reinforcement learning
* Real-time AI-assisted interior design

---

## 🤝 Acknowledgments

This project is developed in collaboration with **Maticad**, which provides both the industrial application context and the professional design datasets used throughout the research and development process.

