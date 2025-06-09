"""Extract document data from CSV and file system for analysis."""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
import logging
from pypdf import PdfReader
import fitz  # PyMuPDF

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.vector_search import VectorSearcher
from src.config import VECTOR_STORE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> dict:
    """Extract text from PDF using multiple methods."""
    
    text_data = {
        "pypdf_text": "",
        "pymupdf_text": "",
        "pages": []
    }
    
    # Try PyPDF first
    try:
        reader = PdfReader(str(pdf_path))
        pypdf_pages = []
        full_text = ""
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            pypdf_pages.append({
                "page_number": i + 1,
                "text": page_text,
                "text_length": len(page_text)
            })
            full_text += page_text + "\n"
        
        text_data["pypdf_text"] = full_text.strip()
        text_data["pypdf_pages"] = pypdf_pages
        text_data["pypdf_page_count"] = len(reader.pages)
        
    except Exception as e:
        logger.warning(f"PyPDF extraction failed for {pdf_path}: {e}")
    
    # Try PyMuPDF for comparison
    try:
        doc = fitz.open(str(pdf_path))
        pymupdf_pages = []
        full_text = ""
        
        for i in range(doc.page_count):
            page = doc[i]
            page_text = page.get_text()
            pymupdf_pages.append({
                "page_number": i + 1,
                "text": page_text,
                "text_length": len(page_text)
            })
            full_text += page_text + "\n"
        
        text_data["pymupdf_text"] = full_text.strip()
        text_data["pymupdf_pages"] = pymupdf_pages
        text_data["pymupdf_page_count"] = doc.page_count
        doc.close()
        
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
    
    # Use PyPDF as primary source
    text_data["pages"] = text_data.get("pypdf_pages", [])
    text_data["full_text"] = text_data.get("pypdf_text", "")
    text_data["page_count"] = text_data.get("pypdf_page_count", 0)
    
    return text_data


def extract_from_csv(csv_path: Path, output_dir: Path):
    """Extract document data from exhibit log CSV."""
    
    documents = []
    
    # Initialize vector search
    vector_search = VectorSearcher(Path(VECTOR_STORE_PATH))
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row['Status'] != 'Success':
                continue
            
            # Extract basic info from CSV
            doc_data = {
                "exhibit_id": row['Exhibit ID'],
                "original_filename": row['Original Filename'],
                "final_filename": row['Final Filename'],
                "category": row['Category'],
                "summary": row['Summary'],
                "bates_range": row['Bates Range'],
                "vector_chunks": int(row['Vector Chunks']) if row['Vector Chunks'] else 0,
                "postgres_stored": row['PostgreSQL Stored'] == 'Yes'
            }
            
            # Find the Bates-numbered PDF
            # First try with underscores
            safe_filename = Path(row['Original Filename']).stem.replace(" ", "_").replace(",", "_")
            bates_pdf_path = output_dir / "bates_numbered" / f"{safe_filename}_BATES.pdf"
            
            if not bates_pdf_path.exists():
                # Try without replacing spaces
                bates_pdf_path = output_dir / "bates_numbered" / f"{Path(row['Original Filename']).stem}_BATES.pdf"
                
            if not bates_pdf_path.exists():
                # Try based on final filename
                bates_pdf_path = output_dir / "bates_numbered" / row['Final Filename'].replace(row['Exhibit ID'] + " - ", "").replace(".pdf", "_BATES.pdf")
            
            # Extract text from PDF
            if bates_pdf_path.exists():
                logger.info(f"Extracting text from: {bates_pdf_path}")
                text_data = extract_text_from_pdf(bates_pdf_path)
                doc_data.update(text_data)
            else:
                logger.warning(f"Bates PDF not found: {bates_pdf_path}")
                doc_data["full_text"] = ""
                doc_data["pages"] = []
                doc_data["page_count"] = 0
            
            # Get vector chunks
            exhibit_num = int(''.join(filter(str.isdigit, row['Exhibit ID'])))
            try:
                # Search for chunks from this exhibit
                search_results = vector_search.search(
                    f"exhibit {exhibit_num}",
                    exhibit_number=exhibit_num,
                    n_results=100  # Get all chunks
                )
                
                vector_chunks = []
                for result in search_results:
                    vector_chunks.append({
                        "chunk_text": result['document'],
                        "metadata": result['metadata'],
                        "score": result.get('score', 0)
                    })
                
                doc_data["vector_chunks_data"] = vector_chunks
                
            except Exception as e:
                logger.warning(f"Could not retrieve vector chunks for exhibit {exhibit_num}: {e}")
                doc_data["vector_chunks_data"] = []
            
            documents.append(doc_data)
    
    return documents


def analyze_document_quality(documents):
    """Analyze extraction quality for documents."""
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_documents": len(documents),
        "documents": []
    }
    
    for doc in documents:
        doc_analysis = {
            "exhibit_id": doc["exhibit_id"],
            "filename": doc["original_filename"],
            "category": doc["category"],
            "metrics": {}
        }
        
        # Text extraction metrics
        text_metrics = {
            "total_characters": len(doc.get("full_text", "")),
            "page_count": doc.get("page_count", 0),
            "pages_analyzed": len(doc.get("pages", [])),
            "average_chars_per_page": 0,
            "empty_pages": [],
            "short_pages": [],
            "extraction_method": "PyPDF"
        }
        
        if text_metrics["page_count"] > 0:
            text_metrics["average_chars_per_page"] = text_metrics["total_characters"] / text_metrics["page_count"]
        
        # Analyze individual pages
        for page in doc.get("pages", []):
            if page["text_length"] == 0:
                text_metrics["empty_pages"].append(page["page_number"])
            elif page["text_length"] < 50:
                text_metrics["short_pages"].append(page["page_number"])
        
        doc_analysis["metrics"]["text_extraction"] = text_metrics
        
        # Chunking metrics
        chunk_metrics = {
            "reported_chunks": doc.get("vector_chunks", 0),
            "actual_chunks": len(doc.get("vector_chunks_data", [])),
            "chunk_sizes": [],
            "average_chunk_size": 0,
            "metadata_completeness": []
        }
        
        if doc.get("vector_chunks_data"):
            chunk_sizes = [len(chunk["chunk_text"]) for chunk in doc["vector_chunks_data"]]
            chunk_metrics["chunk_sizes"] = {
                "min": min(chunk_sizes),
                "max": max(chunk_sizes),
                "average": sum(chunk_sizes) / len(chunk_sizes)
            }
            chunk_metrics["average_chunk_size"] = chunk_metrics["chunk_sizes"]["average"]
            
            # Check metadata
            required_fields = ["exhibit_number", "category", "bates_start", "bates_end"]
            for chunk in doc["vector_chunks_data"]:
                metadata = chunk.get("metadata", {})
                completeness = sum(1 for field in required_fields if field in metadata and metadata[field])
                chunk_metrics["metadata_completeness"].append(completeness / len(required_fields))
        
        doc_analysis["metrics"]["chunking"] = chunk_metrics
        
        # Bates range validation
        bates_metrics = {
            "bates_range": doc.get("bates_range", ""),
            "implied_pages": 0,
            "matches_page_count": False
        }
        
        if doc.get("bates_range"):
            try:
                start, end = doc["bates_range"].split("-")
                start_num = int(''.join(filter(str.isdigit, start)))
                end_num = int(''.join(filter(str.isdigit, end)))
                bates_metrics["implied_pages"] = end_num - start_num + 1
                bates_metrics["matches_page_count"] = bates_metrics["implied_pages"] == doc.get("page_count", 0)
            except:
                pass
        
        doc_analysis["metrics"]["bates_validation"] = bates_metrics
        
        # Overall quality assessment
        quality_score = 0
        quality_issues = []
        
        # Check text extraction
        if text_metrics["total_characters"] > 0:
            quality_score += 1
        else:
            quality_issues.append("No text extracted")
        
        if not text_metrics["empty_pages"]:
            quality_score += 1
        else:
            quality_issues.append(f"{len(text_metrics['empty_pages'])} empty pages")
        
        # Check chunking
        if chunk_metrics["actual_chunks"] > 0:
            quality_score += 1
        else:
            quality_issues.append("No vector chunks found")
        
        if chunk_metrics["actual_chunks"] == chunk_metrics["reported_chunks"]:
            quality_score += 1
        else:
            quality_issues.append("Chunk count mismatch")
        
        # Check Bates
        if bates_metrics["matches_page_count"]:
            quality_score += 1
        else:
            quality_issues.append("Bates range doesn't match page count")
        
        doc_analysis["quality_score"] = quality_score
        doc_analysis["quality_max"] = 5
        doc_analysis["quality_percentage"] = (quality_score / 5) * 100
        doc_analysis["quality_issues"] = quality_issues
        
        if quality_score >= 4:
            doc_analysis["quality_status"] = "excellent"
        elif quality_score >= 3:
            doc_analysis["quality_status"] = "good"
        else:
            doc_analysis["quality_status"] = "needs_review"
        
        analysis["documents"].append(doc_analysis)
    
    # Overall summary
    avg_quality = sum(d["quality_percentage"] for d in analysis["documents"]) / len(analysis["documents"])
    analysis["overall_quality_percentage"] = avg_quality
    
    if avg_quality >= 80:
        analysis["overall_status"] = "excellent"
    elif avg_quality >= 60:
        analysis["overall_status"] = "good"
    else:
        analysis["overall_status"] = "needs_review"
    
    return analysis


def main():
    """Main execution."""
    
    # Find the most recent exhibit log
    exhibit_logs = list(Path("test_improvements").glob("exhibit_log.csv"))
    if not exhibit_logs:
        logger.error("No exhibit log found in test_improvements/")
        return
    
    csv_path = exhibit_logs[0]
    output_dir = csv_path.parent
    
    logger.info(f"Extracting from: {csv_path}")
    
    # Extract document data
    documents = extract_from_csv(csv_path, output_dir)
    
    if not documents:
        logger.error("No successful documents found in CSV")
        return
    
    # Save raw extraction
    exports_dir = project_root / "database_exports"
    exports_dir.mkdir(exist_ok=True)
    
    raw_output = exports_dir / f"csv_extractions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(raw_output, 'w', encoding='utf-8') as f:
        # Don't include full text in initial dump (too large)
        simplified_docs = []
        for doc in documents:
            simplified = doc.copy()
            if "full_text" in simplified:
                simplified["full_text_length"] = len(simplified["full_text"])
                simplified["full_text_preview"] = simplified["full_text"][:500] + "..."
                del simplified["full_text"]
            if "pages" in simplified:
                simplified["page_count"] = len(simplified["pages"])
                del simplified["pages"]
            if "pypdf_text" in simplified:
                del simplified["pypdf_text"]
            if "pymupdf_text" in simplified:
                del simplified["pymupdf_text"]
            if "pypdf_pages" in simplified:
                del simplified["pypdf_pages"]
            if "pymupdf_pages" in simplified:
                del simplified["pymupdf_pages"]
            simplified_docs.append(simplified)
        
        json.dump(simplified_docs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved extraction summary to: {raw_output}")
    
    # Analyze quality
    analysis = analyze_document_quality(documents)
    
    # Save analysis
    analysis_output = exports_dir / f"extraction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Saved analysis to: {analysis_output}")
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DOCUMENT EXTRACTION QUALITY ANALYSIS")
    print("="*80)
    print(f"Documents analyzed: {analysis['total_documents']}")
    print(f"Overall quality: {analysis['overall_status'].upper()} ({analysis['overall_quality_percentage']:.1f}%)")
    
    for doc in analysis["documents"]:
        print(f"\n{'-'*60}")
        print(f"DOCUMENT: {doc['filename']} ({doc['exhibit_id']})")
        print(f"Category: {doc['category']}")
        print(f"Quality: {doc['quality_status'].upper()} ({doc['quality_score']}/{doc['quality_max']})")
        
        if doc['quality_issues']:
            print(f"Issues: {', '.join(doc['quality_issues'])}")
        
        # Text extraction details
        text_m = doc['metrics']['text_extraction']
        print(f"\nText Extraction:")
        print(f"  - Total characters: {text_m['total_characters']:,}")
        print(f"  - Pages: {text_m['page_count']}")
        print(f"  - Avg chars/page: {text_m['average_chars_per_page']:.0f}")
        if text_m['empty_pages']:
            print(f"  - Empty pages: {text_m['empty_pages']}")
        
        # Chunking details
        chunk_m = doc['metrics']['chunking']
        print(f"\nVector Chunking:")
        print(f"  - Chunks: {chunk_m['actual_chunks']} (reported: {chunk_m['reported_chunks']})")
        if chunk_m['chunk_sizes']:
            print(f"  - Chunk sizes: min={chunk_m['chunk_sizes']['min']}, "
                  f"avg={chunk_m['chunk_sizes']['average']:.0f}, "
                  f"max={chunk_m['chunk_sizes']['max']}")
        
        # Bates validation
        bates_m = doc['metrics']['bates_validation']
        print(f"\nBates Validation:")
        print(f"  - Range: {bates_m['bates_range']}")
        print(f"  - Implied pages: {bates_m['implied_pages']}")
        print(f"  - Matches actual: {bates_m['matches_page_count']}")
    
    # Save full text samples for verification
    samples_output = exports_dir / f"text_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    samples = []
    
    for doc in documents[:3]:  # Just the first 3
        sample = {
            "exhibit_id": doc["exhibit_id"],
            "filename": doc["original_filename"],
            "page_samples": []
        }
        
        # Include first and last page text
        if doc.get("pages"):
            if len(doc["pages"]) > 0:
                sample["page_samples"].append({
                    "page": 1,
                    "text_preview": doc["pages"][0]["text"][:500],
                    "full_length": doc["pages"][0]["text_length"]
                })
            if len(doc["pages"]) > 1:
                sample["page_samples"].append({
                    "page": len(doc["pages"]),
                    "text_preview": doc["pages"][-1]["text"][:500],
                    "full_length": doc["pages"][-1]["text_length"]
                })
        
        # Include some vector chunks
        if doc.get("vector_chunks_data"):
            sample["vector_chunk_samples"] = []
            for i, chunk in enumerate(doc["vector_chunks_data"][:3]):
                sample["vector_chunk_samples"].append({
                    "chunk_index": i,
                    "text_preview": chunk["chunk_text"][:300],
                    "metadata": chunk["metadata"]
                })
        
        samples.append(sample)
    
    with open(samples_output, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved text samples to: {samples_output}")


if __name__ == "__main__":
    main()