"""Extract full document data from PostgreSQL and vector store for analysis."""

import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.db_storage import PostgresStorage
from src.vector_search import VectorSearcher
from src.config import POSTGRES_CONNECTION, VECTOR_STORE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_recent_documents(limit: int = 3):
    """Extract data for the most recent documents."""
    
    # Initialize storage connections
    postgres = PostgresStorage(POSTGRES_CONNECTION)
    vector_search = VectorSearcher(Path(VECTOR_STORE_PATH))
    
    try:
        # Get recent documents from PostgreSQL
        query = """
        SELECT DISTINCT 
            d.id,
            d.exhibit_number,
            d.filename,
            d.category,
            d.bates_start,
            d.bates_end,
            d.full_text,
            d.page_count,
            d.created_at,
            COUNT(p.id) as actual_pages
        FROM documents d
        LEFT JOIN pages p ON d.id = p.document_id
        GROUP BY d.id, d.exhibit_number, d.filename, d.category, 
                 d.bates_start, d.bates_end, d.full_text, d.page_count, d.created_at
        ORDER BY d.created_at DESC
        LIMIT %s
        """
        
        with postgres.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                documents = cur.fetchall()
                
                results = []
                for doc in documents:
                    doc_id, exhibit_num, filename, category, bates_start, bates_end, full_text, page_count, created_at, actual_pages = doc
                    
                    # Get page-level data
                    page_query = """
                    SELECT page_number, text, bates_number
                    FROM pages
                    WHERE document_id = %s
                    ORDER BY page_number
                    """
                    cur.execute(page_query, (doc_id,))
                    pages = cur.fetchall()
                    
                    # Get vector chunks for this document
                    vector_results = []
                    if exhibit_num:
                        try:
                            # Search for chunks from this exhibit
                            search_results = vector_search.search(
                                f"exhibit {exhibit_num}",
                                filter={"exhibit_number": exhibit_num},
                                k=100  # Get all chunks for this exhibit
                            )
                            
                            for result in search_results:
                                vector_results.append({
                                    "chunk_text": result['document'],
                                    "metadata": result['metadata'],
                                    "score": result.get('score', 0)
                                })
                        except Exception as e:
                            logger.warning(f"Could not retrieve vector chunks for exhibit {exhibit_num}: {e}")
                    
                    # Build document data
                    doc_data = {
                        "document_id": doc_id,
                        "exhibit_number": exhibit_num,
                        "filename": filename,
                        "category": category,
                        "bates_range": f"{bates_start}-{bates_end}",
                        "page_count": page_count,
                        "actual_page_count": actual_pages,
                        "created_at": created_at.isoformat() if created_at else None,
                        "full_text": {
                            "length": len(full_text) if full_text else 0,
                            "preview": full_text[:500] if full_text else None,
                            "full": full_text
                        },
                        "pages": [
                            {
                                "page_number": page_num,
                                "bates_number": bates_num,
                                "text_length": len(text) if text else 0,
                                "text_preview": text[:200] if text else None,
                                "full_text": text
                            }
                            for page_num, text, bates_num in pages
                        ],
                        "vector_chunks": {
                            "count": len(vector_results),
                            "chunks": vector_results
                        }
                    }
                    
                    results.append(doc_data)
                
                return results
                
    finally:
        postgres.close()


def analyze_extraction_quality(documents):
    """Analyze the quality of text extraction and chunking."""
    
    analysis = {
        "summary": {
            "total_documents": len(documents),
            "timestamp": datetime.now().isoformat()
        },
        "documents": []
    }
    
    for doc in documents:
        doc_analysis = {
            "filename": doc["filename"],
            "exhibit_number": doc["exhibit_number"],
            "category": doc["category"],
            "quality_metrics": {
                "text_extraction": analyze_text_extraction(doc),
                "chunking": analyze_chunking(doc),
                "page_consistency": analyze_page_consistency(doc),
                "vector_indexing": analyze_vector_indexing(doc)
            }
        }
        analysis["documents"].append(doc_analysis)
    
    # Overall statistics
    analysis["summary"]["overall_quality"] = calculate_overall_quality(analysis["documents"])
    
    return analysis


def analyze_text_extraction(doc):
    """Analyze text extraction quality."""
    
    full_text = doc["full_text"]["full"]
    pages = doc["pages"]
    
    metrics = {
        "status": "unknown",
        "total_characters": doc["full_text"]["length"],
        "total_pages": doc["page_count"],
        "pages_with_text": sum(1 for p in pages if p["text_length"] > 0),
        "average_chars_per_page": doc["full_text"]["length"] / doc["page_count"] if doc["page_count"] > 0 else 0,
        "issues": []
    }
    
    # Check for empty pages
    empty_pages = [p["page_number"] for p in pages if p["text_length"] == 0]
    if empty_pages:
        metrics["issues"].append(f"Empty pages detected: {empty_pages}")
    
    # Check for very short pages (might indicate extraction issues)
    short_pages = [p["page_number"] for p in pages if 0 < p["text_length"] < 50]
    if short_pages:
        metrics["issues"].append(f"Very short pages detected: {short_pages}")
    
    # Check if full text matches concatenated page text
    concatenated_length = sum(p["text_length"] for p in pages)
    if abs(concatenated_length - doc["full_text"]["length"]) > 10:
        metrics["issues"].append(f"Full text length mismatch: {doc['full_text']['length']} vs {concatenated_length} (concatenated)")
    
    # Determine status
    if not metrics["issues"]:
        metrics["status"] = "excellent"
    elif len(metrics["issues"]) == 1:
        metrics["status"] = "good"
    else:
        metrics["status"] = "needs_review"
    
    return metrics


def analyze_chunking(doc):
    """Analyze chunking quality."""
    
    chunks = doc["vector_chunks"]["chunks"]
    chunk_count = doc["vector_chunks"]["count"]
    
    metrics = {
        "status": "unknown",
        "total_chunks": chunk_count,
        "chunks_per_page": chunk_count / doc["page_count"] if doc["page_count"] > 0 else 0,
        "chunk_sizes": [],
        "issues": []
    }
    
    if chunks:
        chunk_lengths = [len(c["chunk_text"]) for c in chunks]
        metrics["chunk_sizes"] = {
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "average": sum(chunk_lengths) / len(chunk_lengths)
        }
        
        # Check for very small chunks
        small_chunks = sum(1 for l in chunk_lengths if l < 100)
        if small_chunks > len(chunks) * 0.2:  # More than 20% small chunks
            metrics["issues"].append(f"High proportion of small chunks: {small_chunks}/{len(chunks)}")
        
        # Check for very large chunks
        large_chunks = sum(1 for l in chunk_lengths if l > 1500)
        if large_chunks > 0:
            metrics["issues"].append(f"Large chunks detected: {large_chunks} chunks > 1500 chars")
    
    # Expected chunks based on text size (roughly 750 chars per chunk)
    expected_chunks = doc["full_text"]["length"] / 750
    if chunk_count < expected_chunks * 0.5:
        metrics["issues"].append(f"Fewer chunks than expected: {chunk_count} vs ~{int(expected_chunks)}")
    elif chunk_count > expected_chunks * 2:
        metrics["issues"].append(f"More chunks than expected: {chunk_count} vs ~{int(expected_chunks)}")
    
    # Determine status
    if not metrics["issues"]:
        metrics["status"] = "excellent"
    elif len(metrics["issues"]) == 1:
        metrics["status"] = "good"
    else:
        metrics["status"] = "needs_review"
    
    return metrics


def analyze_page_consistency(doc):
    """Analyze consistency between reported and actual pages."""
    
    metrics = {
        "status": "unknown",
        "reported_pages": doc["page_count"],
        "actual_pages": doc["actual_page_count"],
        "stored_pages": len(doc["pages"]),
        "issues": []
    }
    
    # Check page count consistency
    if metrics["reported_pages"] != metrics["actual_pages"]:
        metrics["issues"].append(f"Page count mismatch: reported {metrics['reported_pages']} vs actual {metrics['actual_pages']}")
    
    if metrics["stored_pages"] != metrics["reported_pages"]:
        metrics["issues"].append(f"Stored pages mismatch: {metrics['stored_pages']} vs reported {metrics['reported_pages']}")
    
    # Check Bates numbering consistency
    bates_start = doc["bates_range"].split("-")[0]
    bates_end = doc["bates_range"].split("-")[1]
    
    if bates_start and bates_end:
        try:
            # Extract numeric part
            start_num = int(''.join(filter(str.isdigit, bates_start)))
            end_num = int(''.join(filter(str.isdigit, bates_end)))
            bates_pages = end_num - start_num + 1
            
            if bates_pages != metrics["reported_pages"]:
                metrics["issues"].append(f"Bates range inconsistent: implies {bates_pages} pages")
        except:
            metrics["issues"].append("Could not parse Bates numbers")
    
    # Determine status
    if not metrics["issues"]:
        metrics["status"] = "excellent"
    else:
        metrics["status"] = "needs_review"
    
    return metrics


def analyze_vector_indexing(doc):
    """Analyze vector indexing quality."""
    
    chunks = doc["vector_chunks"]["chunks"]
    
    metrics = {
        "status": "unknown",
        "chunks_indexed": len(chunks),
        "metadata_completeness": [],
        "issues": []
    }
    
    if chunks:
        # Check metadata completeness
        required_fields = ["exhibit_number", "category", "bates_start", "bates_end"]
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            completeness = sum(1 for field in required_fields if field in metadata and metadata[field])
            metrics["metadata_completeness"].append(completeness / len(required_fields))
        
        avg_completeness = sum(metrics["metadata_completeness"]) / len(metrics["metadata_completeness"])
        metrics["average_metadata_completeness"] = avg_completeness
        
        if avg_completeness < 0.8:
            metrics["issues"].append(f"Incomplete metadata: {avg_completeness:.1%} complete")
    else:
        metrics["issues"].append("No vector chunks found")
    
    # Determine status  
    if not metrics["issues"]:
        metrics["status"] = "excellent"
    elif metrics["chunks_indexed"] > 0 and len(metrics["issues"]) == 1:
        metrics["status"] = "good"
    else:
        metrics["status"] = "needs_review"
    
    return metrics


def calculate_overall_quality(doc_analyses):
    """Calculate overall quality score."""
    
    status_scores = {
        "excellent": 3,
        "good": 2,
        "needs_review": 1,
        "unknown": 0
    }
    
    total_score = 0
    total_metrics = 0
    
    for doc in doc_analyses:
        for metric_type, metric_data in doc["quality_metrics"].items():
            score = status_scores.get(metric_data["status"], 0)
            total_score += score
            total_metrics += 1
    
    avg_score = total_score / total_metrics if total_metrics > 0 else 0
    
    if avg_score >= 2.5:
        return "excellent"
    elif avg_score >= 1.5:
        return "good"
    else:
        return "needs_review"


def main():
    """Main execution."""
    
    logger.info("Extracting recent document data...")
    
    # Extract data
    documents = extract_recent_documents(limit=3)
    
    if not documents:
        logger.error("No documents found in database")
        return
    
    # Save raw data
    output_dir = project_root / "database_exports"
    output_dir.mkdir(exist_ok=True)
    
    raw_output = output_dir / f"document_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_output, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved raw extraction data to: {raw_output}")
    
    # Analyze quality
    logger.info("Analyzing extraction quality...")
    analysis = analyze_extraction_quality(documents)
    
    # Save analysis
    analysis_output = output_dir / f"extraction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Saved analysis to: {analysis_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION QUALITY ANALYSIS")
    print("="*60)
    print(f"Documents analyzed: {analysis['summary']['total_documents']}")
    print(f"Overall quality: {analysis['summary']['overall_quality'].upper()}")
    print("\nDocument Summary:")
    
    for doc in analysis["documents"]:
        print(f"\n{doc['filename']} (Exhibit {doc['exhibit_number']}):")
        for metric_type, metric_data in doc["quality_metrics"].items():
            status = metric_data["status"]
            issues = len(metric_data.get("issues", []))
            print(f"  - {metric_type}: {status.upper()}", end="")
            if issues > 0:
                print(f" ({issues} issues)")
            else:
                print()


if __name__ == "__main__":
    main()