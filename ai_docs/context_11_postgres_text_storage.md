# PostgreSQL Full Text Storage Implementation Plan

## Overview
This document outlines the implementation plan for adding PostgreSQL storage of extracted full text from processed documents. The design emphasizes simplicity, robustness, and modularity while integrating seamlessly with the existing pipeline.

## Design Principles

### 1. Simplicity
- Single responsibility: One module for all PostgreSQL operations
- Minimal dependencies: Use psycopg2 for direct PostgreSQL access
- Clear interfaces: Simple methods for storing and retrieving text

### 2. Robustness
- Connection pooling for efficiency
- Automatic retry logic for transient failures
- Proper transaction management
- Graceful degradation if PostgreSQL is unavailable

### 3. Modularity
- Pluggable architecture: Can be enabled/disabled via configuration
- Decoupled from existing components
- No changes to core processing logic
- Easy to extend for future needs

## Database Schema

```sql
-- Main table for storing document text
CREATE TABLE document_texts (
    id SERIAL PRIMARY KEY,
    exhibit_id INTEGER NOT NULL,
    original_filename VARCHAR(500) NOT NULL,
    exhibit_filename VARCHAR(500) NOT NULL,
    bates_start VARCHAR(20) NOT NULL,
    bates_end VARCHAR(20) NOT NULL,
    category VARCHAR(100),
    full_text TEXT,
    page_count INTEGER,
    char_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for common queries
    INDEX idx_exhibit_id (exhibit_id),
    INDEX idx_bates_range (bates_start, bates_end),
    INDEX idx_category (category),
    INDEX idx_created_at (created_at)
);

-- Optional: Full-text search index
CREATE INDEX idx_full_text_search ON document_texts 
USING gin(to_tsvector('english', full_text));

-- Optional: Page-level storage for granular access
CREATE TABLE document_pages (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES document_texts(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    page_text TEXT,
    bates_number VARCHAR(20),
    
    INDEX idx_document_pages (document_id, page_number),
    INDEX idx_bates_number (bates_number)
);
```

## Implementation Architecture

### Module Structure
```
src/
├── db_storage.py          # New PostgreSQL storage module
├── config.py              # Add PostgreSQL configuration
├── main.py                # Integrate storage after text extraction
└── vector_processor.py    # Share extracted text with storage
```

### Core Components

#### 1. Database Storage Module (`src/db_storage.py`)
```python
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import logging
from typing import Optional, Dict, List
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class PostgresStorage:
    def __init__(self, connection_string: str, pool_size: int = 5):
        """Initialize PostgreSQL storage with connection pooling"""
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, pool_size, connection_string
        )
        self._ensure_tables()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.connection_pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.connection_pool.putconn(conn)
    
    def _ensure_tables(self):
        """Create tables if they don't exist"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create tables SQL here
                pass
    
    def store_document_text(
        self,
        exhibit_id: int,
        original_filename: str,
        exhibit_filename: str,
        bates_start: str,
        bates_end: str,
        category: str,
        full_text: str,
        page_texts: Optional[List[str]] = None
    ) -> int:
        """Store document text and return document ID"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Insert main document record
                cur.execute("""
                    INSERT INTO document_texts 
                    (exhibit_id, original_filename, exhibit_filename, 
                     bates_start, bates_end, category, full_text, 
                     page_count, char_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    exhibit_id, original_filename, exhibit_filename,
                    bates_start, bates_end, category, full_text,
                    len(page_texts) if page_texts else 1,
                    len(full_text)
                ))
                document_id = cur.fetchone()[0]
                
                # Store individual pages if provided
                if page_texts:
                    for i, page_text in enumerate(page_texts, 1):
                        bates_num = self._calculate_bates_number(
                            bates_start, i - 1
                        )
                        cur.execute("""
                            INSERT INTO document_pages 
                            (document_id, page_number, page_text, bates_number)
                            VALUES (%s, %s, %s, %s)
                        """, (document_id, i, page_text, bates_num))
                
                return document_id
    
    def search_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents using PostgreSQL full-text search"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT exhibit_id, exhibit_filename, category,
                           ts_rank(to_tsvector('english', full_text),
                                  plainto_tsquery('english', %s)) as rank,
                           ts_headline('english', full_text,
                                      plainto_tsquery('english', %s),
                                      'MaxWords=50, MinWords=25') as excerpt
                    FROM document_texts
                    WHERE to_tsvector('english', full_text) @@ 
                          plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (query, query, query, limit))
                
                return [
                    {
                        'exhibit_id': row[0],
                        'exhibit_filename': row[1],
                        'category': row[2],
                        'rank': row[3],
                        'excerpt': row[4]
                    }
                    for row in cur.fetchall()
                ]
    
    def get_document_by_bates(self, bates_number: str) -> Optional[Dict]:
        """Retrieve document by Bates number"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT d.*, p.page_text, p.page_number
                    FROM document_texts d
                    LEFT JOIN document_pages p ON p.document_id = d.id
                    WHERE p.bates_number = %s OR 
                          (d.bates_start <= %s AND d.bates_end >= %s)
                    LIMIT 1
                """, (bates_number, bates_number, bates_number))
                
                row = cur.fetchone()
                if row:
                    # Convert to dict
                    return self._row_to_dict(cur.description, row)
                return None
    
    def close(self):
        """Close all connections in the pool"""
        self.connection_pool.closeall()
```

#### 2. Configuration Updates (`src/config.py`)
```python
# PostgreSQL Configuration
ENABLE_POSTGRES_STORAGE = os.getenv("ENABLE_POSTGRES_STORAGE", "false").lower() == "true"
POSTGRES_CONNECTION = os.getenv(
    "POSTGRES_CONNECTION",
    "postgresql://user:password@localhost:5432/bates_documents"
)
POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "5"))

# Storage options
STORE_PAGE_LEVEL_TEXT = os.getenv("STORE_PAGE_LEVEL_TEXT", "true").lower() == "true"
```

#### 3. Integration Point (`src/main.py`)
```python
# Add after vector processing in process_documents()
if config.ENABLE_POSTGRES_STORAGE:
    try:
        # Store extracted text in PostgreSQL
        logger.info(f"[{exhibit_filename}] Storing text in PostgreSQL")
        
        # Get text from vector processor if available, otherwise extract
        if vector_processor and extracted_text:
            full_text = extracted_text
            page_texts = page_texts if config.STORE_PAGE_LEVEL_TEXT else None
        else:
            # Extract text if not already done
            full_text, page_texts = extract_text_for_storage(
                bates_output_path
            )
        
        document_id = postgres_storage.store_document_text(
            exhibit_id=exhibit_counter,
            original_filename=filename,
            exhibit_filename=exhibit_filename,
            bates_start=bates_start,
            bates_end=bates_end,
            category=category,
            full_text=full_text,
            page_texts=page_texts
        )
        
        logger.info(f"[{exhibit_filename}] Stored in PostgreSQL with ID: {document_id}")
        
    except Exception as e:
        logger.error(f"PostgreSQL storage failed for {filename}: {e}")
        # Continue processing - don't fail the whole pipeline
```

#### 4. Shared Text Extraction
Modify `vector_processor.py` to return extracted text:
```python
def process_document(self, pdf_path: str, metadata: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    """Process document and return (chunk_count, full_text, page_texts)"""
    # Existing extraction logic...
    
    # Return extracted data for reuse
    return len(chunks), full_text, page_texts
```

## Integration Flow

```mermaid
graph TD
    A[PDF Document] --> B[Categorization & Summary]
    B --> C[Bates Numbering]
    C --> D[Exhibit Marking]
    D --> E{Text Extraction}
    E --> F[Vector Processing]
    E --> G[PostgreSQL Storage]
    F --> H[ChromaDB]
    G --> I[document_texts table]
    I --> J[document_pages table]
    
    style G fill:#e1f5fe
    style I fill:#e1f5fe
    style J fill:#e1f5fe
```

## Environment Configuration

### .env.template additions:
```bash
# PostgreSQL Storage Configuration
ENABLE_POSTGRES_STORAGE=false
POSTGRES_CONNECTION=postgresql://user:password@localhost:5432/bates_documents
POSTGRES_POOL_SIZE=5
STORE_PAGE_LEVEL_TEXT=true
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install psycopg2-binary
```

### 2. Create Database
```sql
CREATE DATABASE bates_documents;
```

### 3. Run Schema Creation
```bash
python -c "from src.db_storage import PostgresStorage; PostgresStorage(connection_string).setup_database()"
```

## Search CLI Extension

Add PostgreSQL search option to `src/search_cli.py`:
```python
parser.add_argument(
    '--search-engine',
    choices=['vector', 'postgres', 'both'],
    default='vector',
    help='Search engine to use'
)

# In search logic:
if args.search_engine in ['postgres', 'both']:
    postgres_results = postgres_storage.search_text(
        args.query, 
        args.num_results
    )
    # Display results...
```

## Benefits

1. **Full-Text Search**: Native PostgreSQL FTS capabilities
2. **SQL Queries**: Complex queries across metadata
3. **Page-Level Access**: Retrieve specific pages by Bates number
4. **Backup/Recovery**: Standard PostgreSQL tools
5. **Analytics**: SQL-based reporting and analysis
6. **Scalability**: PostgreSQL handles large datasets well

## Performance Considerations

1. **Batch Inserts**: For multiple documents, use batch operations
2. **Connection Pooling**: Reuse connections efficiently
3. **Index Strategy**: Create indexes based on query patterns
4. **Text Compression**: PostgreSQL automatically compresses TEXT fields
5. **Async Options**: Consider asyncpg for high-throughput scenarios

## Error Handling

1. **Connection Failures**: Retry with exponential backoff
2. **Transaction Rollback**: Ensure data consistency
3. **Graceful Degradation**: Pipeline continues if PostgreSQL fails
4. **Logging**: Detailed error logging for troubleshooting

## Future Enhancements

1. **Metadata Storage**: Store LLM summaries and classifications
2. **Version History**: Track document modifications
3. **Full Pipeline State**: Store complete processing history
4. **Advanced Search**: Integrate with vector search for hybrid queries
5. **REST API**: Expose PostgreSQL data via API endpoints

## Testing Strategy

1. **Unit Tests**: Test each method in PostgresStorage
2. **Integration Tests**: Test with real PDFs
3. **Performance Tests**: Benchmark with large documents
4. **Failure Tests**: Test connection failures and recovery
5. **Search Tests**: Validate full-text search accuracy

## Migration Path

For existing deployments:
1. Enable PostgreSQL storage for new documents
2. Run batch migration script for existing documents
3. Verify data integrity
4. Update search interfaces
5. Monitor performance

This implementation provides a robust, modular solution for storing extracted text in PostgreSQL while maintaining the simplicity and reliability of the existing pipeline.