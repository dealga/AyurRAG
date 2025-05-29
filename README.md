# AyurRAG

This project implements a RAG (Retrieval-Augmented Generation) system for Ayurvedic knowledge.

## Project Overview

This application uses llama and vector databases to retrieve relevant information from Ayurvedic texts and generate accurate responses to user queries about Ayurvedic medicine and practices.

## Key Components

* **Vector Database**: Milvus for efficient similarity search
* **Embedding Model**: Sentence transformers for creating text embeddings
* **Django Backend**: Handles API requests and business logic
* **Frontend**: Interactive user interface

## Setup and Installation

### Prerequisites

* Python 3.12+
* Docker (for Milvus)

### Installation

1. Clone the repository

   ```
   git clone https://github.com/dealga/Ayur-FinalYearProject.git
   cd Ayur-FinalYearProject
   ```

2. Create and activate a virtual environment

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```
   pip install -r requirements.txt
   ```

4. **Create embeddings and insert data**

   * Run `embeddings.py` to create embeddings for your Ayurvedic book/text. Text should be extracted from PDFs beforehand, and stored in .txt files. 
     **Make sure to update the file path inside `embeddings.py` to point to your required book/text file before running.**

     ```
     python embeddings.py
     ```
   * After embeddings are created, run `insertion.py` to push these embeddings to Milvus and the corresponding sentences to SQLite. Both share a common ID to maintain consistency.

     ```
     python insertion.py
     ```

5. Set up the database

   ```
   cd AyurGPT
   python manage.py migrate
   ```

6. Start the server

   ```
   python manage.py runserver
   ```

## Usage

\[Add usage instructions here]

## Project Structure

\[Add a brief description of your project structure here]

## License

\[Add license information here]

## Contributors

* Dhanush H
* Chandan Gowda T K
* Srivatsa G
