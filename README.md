# LLM Chat RAG - Web Interface

## Overview

The **LLM Chat RAG** (Retriever-Augmented Generation) is a web application that allows you to interact with documents stored in a ChromaDB database using OpenAI's GPT-4o-mini model. This system provides a way to query documents for relevant information, retrieve context, and generate AI-driven responses based on that context.

## Features

- **Retriever-Augmented Generation (RAG):** Retrieve context from a database and use it to generate more accurate and informative responses.
- **OpenAI GPT-4o-mini Model:** Uses OpenAI's GPT-4o-mini to generate answers based on the retrieved context.
- **ChromaDB Integration:** Uses ChromaDB for efficient document retrieval and context management.
- **Web-based UI:** Interact with the system through a user-friendly web interface.

## Prerequisites

- **Python 3.x**: Ensure Python 3.6 or later is installed.
- **OpenAI API Key**: You must have an OpenAI API key to use the GPT model.
- **ChromaDB**: A local or cloud-based ChromaDB instance for document storage and retrieval.

## Setup

### 1. Install Dependencies

Install required dependencies via `pip`:

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Set the following environment variables:
``` bash
OPENAI_API_KEY=your_openai_api_key
CHROMA_DB_PATH=./chroma_db  # (Optional) Path to the ChromaDB persistent database
```

For example, on Linux or macOS:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export CHROMA_DB_PATH="./chroma_db"
```

On Windows:
```bash
set OPENAI_API_KEY=your_openai_api_key
set CHROMA_DB_PATH=.\chroma_db
```

### 3. Run the Application

To start the web application, run the following command:

```bash
python3 main.py
```

Then open your browser and navigate to the local URL (e.g., http://localhost:5000) to access the chat interface.

### 4. Setup ChromaDB (Optional)

If you need to set up the ChromaDB collection before running the application, run:

```bash
python3 main.py --setup
```

This will initialize the ChromaDB collection with OpenAI embeddings.

## Web Interface

Once the application is running, the web UI will present you with:
- An input field to type your query.
- A display area to show the generated response along with the sources used.
- Navigation options to access additional functionalities (e.g., viewing past interactions).

## How it Works

- **ChromaDB Setup:** On running the script, it will either set up a new ChromaDB collection or connect to an existing one containing document embeddings.
- **Retriever-Augmented Generation (RAG):** When a user submits a question via the web UI, the system retrieves relevant documents from ChromaDB and uses the retrieved context along with OpenAI's GPT model to generate an answer.
- **Response Generation:** The generated response is displayed on the web page, along with any sources used.

## Example Usage

1. Navigate to your web browser (e.g., http://localhost:5000).
2. Type your question in the provided input field and submit.
3. View the generated response along with any source references.

## Development

To contribute to this project, feel free to fork the repository, make changes, and create pull requests.

## License

This project is licensed under the GNU General Public License - see the LICENSE file for details.
