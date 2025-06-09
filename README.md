# AI-Powered Bates Numbering & Exhibit Marking Tool

This tool automates the process of Bates numbering and exhibit marking for legal documents using AI-powered categorization, vector search, and full-text storage capabilities.

## Features

- **Automated Bates Numbering**: Applies sequential Bates numbers to PDF documents
- **AI-Powered Categorization**: Uses LLMs to categorize documents into legal categories
- **Exhibit Marking**: Automatically marks documents with exhibit numbers based on categories
- **Vector Search**: Semantic search across all processed documents using embeddings
- **Full-Text Storage**: PostgreSQL storage for full-text search capabilities
- **CLI Tools**: Command-line interfaces for processing and searching documents
- **Multi-Model Support**: Supports OpenAI, Ollama, LM Studio (MLX models), and AWS Bedrock for vision and language tasks

### Document Categories
- Pleading
- Medical Record
- Bill
- Correspondence
- Photo
- Video
- Documentary Evidence
- Uncategorized

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, for full-text search)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bates-number-demo.git
   cd bates-number-demo
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.template .env
   ```
   Edit `.env` and configure:
   - **For OpenAI**: Set `OPENAI_API_KEY` and `LLM_PROVIDER=openai`
   - **For Ollama**: Set `LLM_PROVIDER=ollama` and run `python setup_ollama.py`
   - **For LM Studio**: Set `LLM_PROVIDER=lmstudio` (see LMSTUDIO_SETUP.md)
   - `ENABLE_POSTGRES_STORAGE`: Set to "true" for PostgreSQL support
   - `POSTGRES_CONNECTION`: PostgreSQL connection string
   - Other optional settings

5. **Set up PostgreSQL (optional)**
   ```bash
   createdb bates_documents
   ```

## Usage

### Basic Document Processing

Place PDF documents in the `input_documents/` directory, then run:

```bash
python src/main.py
```

### Command-Line Options

```bash
python src/main.py [options]
```

Options:
- `--input_dir PATH`: Input directory (default: `input_documents/`)
- `--output_dir PATH`: Output directory (default: `output/`)
- `--llm_model MODEL`: LLM model for categorization
- `--exhibit_prefix PREFIX`: Exhibit mark prefix (default: "Exhibit ")
- `--bates_prefix PREFIX`: Bates number prefix (e.g., "ABC")
- `--bates_digits N`: Number of digits for Bates numbers (default: 6)

### Document Search

Search processed documents using semantic search:

```bash
# Basic search
python src/search_cli.py "motion to dismiss"

# Search within a category
python src/search_cli.py "patient treatment" --category "Medical Record"

# Get more results
python src/search_cli.py "insurance coverage" -n 20

# Use PostgreSQL full-text search
python src/search_cli.py "payment receipt" --search-engine postgres

# Compare both search engines
python src/search_cli.py "legal precedent" --search-engine both
```

Search options:
- `-n, --num-results`: Number of results to return
- `-c, --category`: Filter by document category
- `-e, --exhibit`: Filter by exhibit number
- `--bates-start/--bates-end`: Search by Bates range
- `--search-engine`: Choose vector, postgres, or both
- `--stats`: Show database statistics
- `--categories`: List available categories

## Output Structure

```
output/
├── bates_numbered/          # Bates-numbered PDFs
├── exhibits/                # Exhibit-marked PDFs organized by category
│   ├── bill/
│   ├── correspondence/
│   ├── documentary_evidence/
│   ├── medical_record/
│   ├── photo/
│   ├── pleading/
│   ├── uncategorized/
│   └── video/
└── exhibit_log.csv         # Processing log with metadata
```

## Architecture

The system consists of several key components:

1. **PDF Processor** (`pdf_processor.py`): Handles PDF manipulation and stamping
2. **LLM Handler** (`llm_handler.py`): Manages AI categorization and naming
3. **Vector Processor** (`vector_processor.py`): Creates embeddings for search
4. **Database Storage** (`db_storage.py`): PostgreSQL full-text storage
5. **Search Interface** (`search_cli.py`): Command-line search tool

## Configuration

Key environment variables in `.env`:

```bash
# LLM Provider
LLM_PROVIDER=openai  # or ollama, lmstudio, bedrock

# OpenAI
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini-2024-07-18

# LM Studio (for Apple Silicon MLX models)
LMSTUDIO_HOST=http://localhost:1234/v1
LMSTUDIO_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit

# PostgreSQL (optional)
ENABLE_POSTGRES_STORAGE=true
POSTGRES_CONNECTION=postgresql://user:pass@localhost:5432/bates_documents

# Vector Store
ENABLE_VECTOR_STORE=true
VECTOR_STORE_PATH=vector_store

# Processing
BATCH_SIZE=1
MAX_RETRIES=3
```

## Development

### Project Structure
```
├── src/                    # Source code
│   ├── main.py            # Main processing script
│   ├── pdf_processor.py   # PDF manipulation
│   ├── llm_handler.py     # AI integration
│   ├── vector_processor.py # Embeddings
│   ├── db_storage.py      # PostgreSQL storage
│   ├── search_cli.py      # Search interface
│   └── config.py          # Configuration
├── ai_docs/               # AI context documentation
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
└── .env.template         # Environment template
```

### Testing

Run tests with pytest:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with PyMuPDF for PDF processing
- Uses OpenAI GPT models for categorization
- ChromaDB for vector storage
- PostgreSQL for full-text search