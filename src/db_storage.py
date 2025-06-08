"""PostgreSQL storage module for document text and metadata."""

import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from contextlib import contextmanager
import logging
from typing import Optional, Dict, List, Tuple, Any
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class PostgresStorage:
    """Handles PostgreSQL storage for document texts with connection pooling."""
    
    def __init__(self, connection_string: str, pool_size: int = 5):
        """Initialize PostgreSQL storage with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string
            pool_size: Maximum number of connections in the pool
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.connection_pool = None
        self._initialize_pool()
        self._ensure_tables()
    
    def _initialize_pool(self):
        """Initialize the connection pool with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1, self.pool_size, self.connection_string
                )
                logger.info(f"PostgreSQL connection pool initialized (size: {self.pool_size})")
                return
            except psycopg2.OperationalError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create connection pool after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        create_tables_sql = """
        -- Main table for storing document text
        CREATE TABLE IF NOT EXISTS document_texts (
            id SERIAL PRIMARY KEY,
            exhibit_id INTEGER NOT NULL UNIQUE,
            original_filename VARCHAR(500) NOT NULL,
            exhibit_filename VARCHAR(500) NOT NULL,
            bates_start VARCHAR(20) NOT NULL,
            bates_end VARCHAR(20) NOT NULL,
            category VARCHAR(100),
            full_text TEXT,
            page_count INTEGER,
            char_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_exhibit_id ON document_texts(exhibit_id);
        CREATE INDEX IF NOT EXISTS idx_bates_range ON document_texts(bates_start, bates_end);
        CREATE INDEX IF NOT EXISTS idx_category ON document_texts(category);
        CREATE INDEX IF NOT EXISTS idx_created_at ON document_texts(created_at);
        
        -- Full-text search index
        CREATE INDEX IF NOT EXISTS idx_full_text_search ON document_texts 
        USING gin(to_tsvector('english', full_text));
        
        -- Page-level storage for granular access
        CREATE TABLE IF NOT EXISTS document_pages (
            id SERIAL PRIMARY KEY,
            document_id INTEGER REFERENCES document_texts(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            page_text TEXT,
            bates_number VARCHAR(20),
            UNIQUE(document_id, page_number)
        );
        
        -- Indexes for page-level queries
        CREATE INDEX IF NOT EXISTS idx_document_pages ON document_pages(document_id, page_number);
        CREATE INDEX IF NOT EXISTS idx_bates_number ON document_pages(bates_number);
        
        -- Update trigger for updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_document_texts_updated_at ON document_texts;
        CREATE TRIGGER update_document_texts_updated_at 
        BEFORE UPDATE ON document_texts 
        FOR EACH ROW 
        EXECUTE FUNCTION update_updated_at_column();
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_tables_sql)
                    logger.info("PostgreSQL tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
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
        """Store document text and return document ID.
        
        Args:
            exhibit_id: Exhibit number
            original_filename: Original PDF filename
            exhibit_filename: Generated exhibit filename
            bates_start: Starting Bates number
            bates_end: Ending Bates number
            category: Document category
            full_text: Complete document text
            page_texts: Optional list of text per page
            
        Returns:
            Document ID from database
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Upsert main document record
                cur.execute("""
                    INSERT INTO document_texts 
                    (exhibit_id, original_filename, exhibit_filename, 
                     bates_start, bates_end, category, full_text, 
                     page_count, char_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exhibit_id) 
                    DO UPDATE SET
                        original_filename = EXCLUDED.original_filename,
                        exhibit_filename = EXCLUDED.exhibit_filename,
                        bates_start = EXCLUDED.bates_start,
                        bates_end = EXCLUDED.bates_end,
                        category = EXCLUDED.category,
                        full_text = EXCLUDED.full_text,
                        page_count = EXCLUDED.page_count,
                        char_count = EXCLUDED.char_count,
                        updated_at = CURRENT_TIMESTAMP
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
                    # Delete existing pages for this document
                    cur.execute("DELETE FROM document_pages WHERE document_id = %s", (document_id,))
                    
                    # Batch insert pages
                    page_data = []
                    for i, page_text in enumerate(page_texts, 1):
                        bates_num = self._calculate_bates_number(bates_start, i - 1)
                        page_data.append((document_id, i, page_text, bates_num))
                    
                    extras.execute_batch(
                        cur,
                        """
                        INSERT INTO document_pages 
                        (document_id, page_number, page_text, bates_number)
                        VALUES (%s, %s, %s, %s)
                        """,
                        page_data,
                        page_size=100
                    )
                
                logger.info(f"Stored document {exhibit_id} with {len(page_texts) if page_texts else 1} pages")
                return document_id
    
    def _calculate_bates_number(self, bates_start: str, offset: int) -> str:
        """Calculate Bates number with offset from start."""
        # Extract prefix and number
        prefix = ""
        number_part = bates_start
        
        # Find where the number starts
        for i, char in enumerate(bates_start):
            if char.isdigit():
                prefix = bates_start[:i]
                number_part = bates_start[i:]
                break
        
        # Calculate new number
        start_num = int(number_part)
        new_num = start_num + offset
        
        # Format with same number of digits
        return f"{prefix}{str(new_num).zfill(len(number_part))}"
    
    def search_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using PostgreSQL full-text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results with metadata and excerpts
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        exhibit_id, 
                        exhibit_filename, 
                        category,
                        bates_start,
                        bates_end,
                        ts_rank(to_tsvector('english', full_text),
                               plainto_tsquery('english', %s)) as rank,
                        ts_headline('english', full_text,
                                   plainto_tsquery('english', %s),
                                   'MaxWords=50, MinWords=25, StartSel=<b>, StopSel=</b>') as excerpt
                    FROM document_texts
                    WHERE to_tsvector('english', full_text) @@ 
                          plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (query, query, query, limit))
                
                results = []
                for row in cur.fetchall():
                    results.append(dict(row))
                
                return results
    
    def get_document_by_bates(self, bates_number: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by Bates number.
        
        Args:
            bates_number: Bates number to search for
            
        Returns:
            Document data with page text if found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                # First try to find by exact page match
                cur.execute("""
                    SELECT 
                        d.*,
                        p.page_text,
                        p.page_number
                    FROM document_texts d
                    JOIN document_pages p ON p.document_id = d.id
                    WHERE p.bates_number = %s
                """, (bates_number,))
                
                row = cur.fetchone()
                if row:
                    return dict(row)
                
                # If not found, try range search
                cur.execute("""
                    SELECT *
                    FROM document_texts
                    WHERE %s >= bates_start AND %s <= bates_end
                    LIMIT 1
                """, (bates_number, bates_number))
                
                row = cur.fetchone()
                if row:
                    return dict(row)
                
                return None
    
    def get_document_by_exhibit(self, exhibit_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve document by exhibit ID.
        
        Args:
            exhibit_id: Exhibit ID to retrieve
            
        Returns:
            Document data if found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM document_texts
                    WHERE exhibit_id = %s
                """, (exhibit_id,))
                
                row = cur.fetchone()
                return dict(row) if row else None
    
    def get_pages_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Retrieve all pages for a document.
        
        Args:
            document_id: Database document ID
            
        Returns:
            List of page data
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM document_pages
                    WHERE document_id = %s
                    ORDER BY page_number
                """, (document_id,))
                
                return [dict(row) for row in cur.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with counts and statistics
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                # Document counts by category
                cur.execute("""
                    SELECT 
                        category,
                        COUNT(*) as count,
                        SUM(page_count) as total_pages,
                        SUM(char_count) as total_chars
                    FROM document_texts
                    GROUP BY category
                    ORDER BY count DESC
                """)
                categories = [dict(row) for row in cur.fetchall()]
                
                # Overall statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(page_count) as total_pages,
                        SUM(char_count) as total_chars,
                        MIN(created_at) as first_document,
                        MAX(created_at) as last_document
                    FROM document_texts
                """)
                overall = dict(cur.fetchone())
                
                return {
                    'overall': overall,
                    'by_category': categories
                }
    
    def close(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")