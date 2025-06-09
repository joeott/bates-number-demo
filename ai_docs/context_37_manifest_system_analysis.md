# Context 37: PDF Manifest System Analysis

## Problem Statement

The current system lacks a comprehensive way to track and manage PDF files across input folders before processing. This creates several challenges:
- No clear inventory of what needs to be processed
- Difficulty tracking processing status
- Risk of duplicate processing
- No clear audit trail from source to output
- Hard to resume after failures

## Deep Analysis

### Core Requirements

1. **Discovery**: Find all PDF files in nested folder structures
2. **Identification**: Assign unique IDs to track files through pipeline
3. **Registration**: Store file metadata in database before processing
4. **Status Tracking**: Know what's been processed, what's pending
5. **Relationship Management**: Link source files to their outputs

### Architectural Considerations

#### Why This Is More Complex Than It Appears

1. **File Identity Problem**
   - Same file might exist in multiple locations
   - Files might be moved/renamed during processing
   - Need to detect duplicates based on content, not just name
   - Hash-based identification vs path-based

2. **State Management**
   - Processing pipeline has multiple stages
   - Each stage might fail independently
   - Need to track: discovered → registered → processed → output generated
   - Partial processing scenarios

3. **Scalability Concerns**
   - Folders might contain thousands of PDFs
   - Need efficient batch operations
   - Memory constraints for large manifests
   - Database transaction sizes

4. **Metadata Richness**
   - Basic: filename, path, size, modified date
   - Advanced: page count, PDF metadata, hash
   - Extracted: document type, exhibit number, Bates range
   - Relationships: parent folder, exhibit group

5. **Error Recovery**
   - Network/disk failures during discovery
   - Corrupted PDFs
   - Database connection issues
   - Need graceful degradation

## Proposed Solution: Document Manifest System

### Database Schema

```sql
-- Document manifest table
CREATE TABLE document_manifest (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_size BIGINT,
    file_hash TEXT,  -- SHA256 of file content
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'discovered',  -- discovered, registered, processing, processed, failed
    error_message TEXT,
    metadata JSONB,  -- Flexible field for additional data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing history table
CREATE TABLE processing_history (
    id SERIAL PRIMARY KEY,
    manifest_id UUID REFERENCES document_manifest(id),
    stage TEXT NOT NULL,  -- ingestion, vector_store, bates_numbering, etc.
    status TEXT NOT NULL,  -- started, completed, failed
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_details TEXT,
    output_path TEXT
);

-- Indexes for performance
CREATE INDEX idx_manifest_status ON document_manifest(status);
CREATE INDEX idx_manifest_hash ON document_manifest(file_hash);
CREATE INDEX idx_processing_manifest ON processing_history(manifest_id);
```

### Implementation Strategy

#### Phase 1: Discovery Tool

```python
class PDFManifestBuilder:
    """Discovers and catalogs PDF files in directory structures."""
    
    def discover_pdfs(self, root_path: Path) -> List[Dict]:
        """
        Recursively find all PDFs and extract metadata.
        Returns list of dicts with file information.
        """
        # Use glob for efficiency
        # Calculate file hashes for deduplication
        # Extract basic PDF metadata
        # Return structured data
        
    def generate_manifest_json(self, pdf_list: List[Dict]) -> str:
        """Generate JSON manifest for discovered PDFs."""
        # Include discovery timestamp
        # Group by folder structure
        # Add summary statistics
```

#### Phase 2: Registration Tool

```python
class DocumentRegistrar:
    """Registers documents in the database with tracking."""
    
    def register_documents(self, manifest: List[Dict]) -> Dict[str, UUID]:
        """
        Register documents in database, return mapping of paths to UUIDs.
        Handles duplicates gracefully.
        """
        # Batch insert for efficiency
        # Check for existing entries by hash
        # Update status for re-discovered files
        # Return path->UUID mapping
        
    def mark_for_processing(self, document_ids: List[UUID]):
        """Mark documents as ready for processing pipeline."""
```

#### Phase 3: Status Tracking

```python
class ProcessingTracker:
    """Tracks document processing through pipeline stages."""
    
    def start_processing(self, manifest_id: UUID, stage: str):
        """Record start of processing stage."""
        
    def complete_processing(self, manifest_id: UUID, stage: str, output_path: str = None):
        """Record successful completion."""
        
    def fail_processing(self, manifest_id: UUID, stage: str, error: str):
        """Record processing failure."""
        
    def get_pending_documents(self, stage: str = None) -> List[Dict]:
        """Get documents pending processing."""
```

### Integration Points

1. **CLI Command**
   ```bash
   python src/manifest_builder.py scan --input-dir /path/to/pdfs
   python src/manifest_builder.py status
   python src/manifest_builder.py reset --confirm
   ```

2. **Main Pipeline Integration**
   - DocumentOrchestrator checks manifest before processing
   - Updates manifest status after each stage
   - Can resume from manifest state

3. **Monitoring**
   - Web dashboard showing processing status
   - Export capabilities for reporting
   - Audit trail for compliance

### Implementation Simplifications

For immediate implementation, focus on:

1. **Simple File Scanner**
   ```python
   def scan_for_pdfs(root_dir: Path) -> Dict[str, Any]:
       """Simple PDF discovery returning JSON."""
       pdfs = []
       for pdf_path in Path(root_dir).rglob("*.pdf"):
           pdfs.append({
               "id": str(uuid.uuid4()),
               "path": str(pdf_path),
               "name": pdf_path.name,
               "size": pdf_path.stat().st_size,
               "modified": pdf_path.stat().st_mtime
           })
       return {
           "scan_date": datetime.now().isoformat(),
           "root_directory": str(root_dir),
           "total_files": len(pdfs),
           "documents": pdfs
       }
   ```

2. **Simple Database Table**
   ```sql
   CREATE TABLE simple_manifest (
       id UUID PRIMARY KEY,
       file_path TEXT UNIQUE NOT NULL,
       file_name TEXT NOT NULL,
       status TEXT DEFAULT 'pending',
       created_at TIMESTAMP DEFAULT NOW()
   );
   ```

3. **Minimal Status Updates**
   - Just track: pending → processing → complete/failed
   - Store in PostgreSQL alongside existing tables
   - Simple queries for pending work

### Benefits of This Approach

1. **Clear Inventory**: Know exactly what needs processing
2. **Resume Capability**: Pick up where left off after failures
3. **Deduplication**: Avoid processing same file twice
4. **Audit Trail**: Complete history of what was processed when
5. **Scalability**: Can handle large document sets efficiently
6. **Flexibility**: JSONB metadata field for future needs

### Next Steps

1. Implement simple scanner function in utils.py
2. Add manifest table to database schema
3. Modify DocumentOrchestrator to check manifest
4. Add CLI commands for manifest operations
5. Create status reporting functionality

This manifest system provides the foundation for robust, resumable document processing while maintaining simplicity in implementation.