#!/usr/bin/env python3
"""
Command-line interface for searching legal documents using vector search.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure src directory is in Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.vector_search import VectorSearcher
from src.utils import setup_logging
from src.config import VECTOR_STORE_PATH, ENABLE_POSTGRES_STORAGE, POSTGRES_CONNECTION, POSTGRES_POOL_SIZE
from src.db_storage import PostgresStorage
from src.hybrid_search import HybridSearcher, SearchMethod


def format_bates_range(start: int, end: int) -> str:
    """Format Bates range for display."""
    if start == end:
        return f"{start:06d}"
    return f"{start:06d}-{end:06d}"


def display_results(results, verbose: bool = False, search_type: str = "vector"):
    """Display search results in a formatted manner."""
    if not results:
        print("\nNo results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        
        # Handle both SearchResult objects and dictionary results
        if hasattr(result, 'document_id'):
            # SearchResult object from hybrid search
            print(f"Document: {result.filename}")
            print(f"Category: {result.category}")
            print(f"Exhibit #: {result.exhibit_number}")
            print(f"Bates: {result.bates_range}")
            if hasattr(result, 'page') and result.page:
                print(f"Page: {result.page}")
            print(f"Score: {result.score:.4f}")
            if result.source:
                print(f"Source: {result.source}")
            
            # Show excerpt
            text = result.text
            max_excerpt_length = 300 if not verbose else 500
            if len(text) > max_excerpt_length:
                excerpt = text[:max_excerpt_length] + "..."
            else:
                excerpt = text
            print(f"Excerpt: {excerpt}")
            
        elif search_type == "postgres":
            # PostgreSQL result format
            print(f"Document: {result['exhibit_filename']}")
            print(f"Category: {result['category']}")
            print(f"Exhibit #: {result['exhibit_id']}")
            print(f"Bates: {result['bates_start']}-{result['bates_end']}")
            
            if 'rank' in result:
                print(f"Relevance: {result['rank']:.4f}")
            
            if 'excerpt' in result:
                # PostgreSQL highlights with <b> tags
                excerpt = result['excerpt'].replace('<b>', '').replace('</b>', '')
                print(f"Excerpt: {excerpt}")
        else:
            # Vector search result format
            print(f"Document: {result['filename']}")
            print(f"Category: {result['category']}")
            print(f"Exhibit #: {result['exhibit_number']}")
            print(f"Bates: {format_bates_range(result['bates_start'], result['bates_end'])}")
            print(f"Page: {result['page']}")
            
            if 'relevance' in result:
                print(f"Relevance: {result['relevance']:.2%}")
            
            if result.get('summary'):
                print(f"Summary: {result['summary']}")
            
            # Show excerpt
            text = result['text']
            max_excerpt_length = 300 if not verbose else 500
            if len(text) > max_excerpt_length:
                excerpt = text[:max_excerpt_length] + "..."
            else:
                excerpt = text
            print(f"Excerpt: {excerpt}")
        
        print()  # Blank line between results


def main():
    parser = argparse.ArgumentParser(
        description="Search legal documents using semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python src/search_cli.py "motion to dismiss"
  
  # Search within a category
  python src/search_cli.py "patient treatment" --category "Medical Record"
  
  # Get more results
  python src/search_cli.py "insurance coverage" -n 20
  
  # Search by Bates range
  python src/search_cli.py --bates-start 100 --bates-end 200
  
  # Show statistics
  python src/search_cli.py --stats
""")
    
    # Search arguments
    parser.add_argument("query", nargs="?", help="Search query text")
    parser.add_argument("-n", "--num-results", type=int, default=10,
                        help="Number of results to return (default: 10)")
    parser.add_argument("-c", "--category", type=str,
                        help="Filter by document category")
    parser.add_argument("-e", "--exhibit", type=int,
                        help="Filter by exhibit number")
    parser.add_argument("--bates-start", type=int,
                        help="Search by Bates range start")
    parser.add_argument("--bates-end", type=int,
                        help="Search by Bates range end")
    
    # Display options
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show more detailed results")
    parser.add_argument("--stats", action="store_true",
                        help="Show vector store statistics")
    parser.add_argument("--categories", action="store_true",
                        help="List available categories")
    
    # Search engine selection
    parser.add_argument("--search-engine", choices=["vector", "postgres", "hybrid", "both"],
                        default="vector", help="Search engine to use")
    
    # Path options
    parser.add_argument("--vector-store", type=Path, default=project_root / VECTOR_STORE_PATH,
                        help="Path to vector store")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.query and not args.stats and not args.categories and not (args.bates_start or args.bates_end):
        parser.error("Please provide a search query or use --stats/--categories/--bates-range")
    
    if (args.bates_start is not None) != (args.bates_end is not None):
        parser.error("Both --bates-start and --bates-end must be provided together")
    
    # Setup logging
    setup_logging()
    
    # Initialize search components based on engine selection
    vector_searcher = None
    postgres_searcher = None
    hybrid_searcher = None
    
    if args.search_engine in ["vector", "both"]:
        try:
            vector_searcher = VectorSearcher(str(args.vector_store))
        except Exception as e:
            print(f"Error: Failed to initialize vector search: {e}")
            if args.search_engine == "vector":
                sys.exit(1)
    
    if args.search_engine in ["postgres", "both"]:
        if not ENABLE_POSTGRES_STORAGE:
            print("Error: PostgreSQL storage is not enabled. Set ENABLE_POSTGRES_STORAGE=true in .env")
            sys.exit(1)
        try:
            postgres_searcher = PostgresStorage(
                connection_string=POSTGRES_CONNECTION,
                pool_size=POSTGRES_POOL_SIZE
            )
        except Exception as e:
            print(f"Error: Failed to initialize PostgreSQL search: {e}")
            if args.search_engine == "postgres":
                sys.exit(1)
    
    if args.search_engine == "hybrid":
        try:
            hybrid_searcher = HybridSearcher(
                vector_store_path=str(args.vector_store),
                postgres_config={
                    'connection_string': POSTGRES_CONNECTION,
                    'pool_size': POSTGRES_POOL_SIZE
                } if ENABLE_POSTGRES_STORAGE else None
            )
        except Exception as e:
            print(f"Error: Failed to initialize hybrid search: {e}")
            sys.exit(1)
    
    # Handle different operations
    if args.stats:
        if hybrid_searcher:
            stats = hybrid_searcher.get_statistics()
            print("\n=== Hybrid Search Statistics ===")
            if 'vector' in stats:
                print("Vector Store:")
                print(f"  Total chunks: {stats['vector'].get('total_chunks', 0)}")
                print(f"  Categories: {stats['vector'].get('num_categories', 0)}")
                if stats['vector'].get('categories'):
                    print(f"  Available categories: {', '.join(stats['vector']['categories'])}")
            
            if 'postgres' in stats:
                print("PostgreSQL:")
                print(f"  Total documents: {stats['postgres']['overall']['total_documents']}")
                print(f"  Total pages: {stats['postgres']['overall']['total_pages']}")
                print(f"  Total characters: {stats['postgres']['overall']['total_chars']:,}")
                print("  Documents by category:")
                for cat in stats['postgres']['by_category']:
                    print(f"    {cat['category']}: {cat['count']} documents, {cat['total_pages']} pages")
        
        elif vector_searcher:
            stats = vector_searcher.get_stats()
            print("\n=== Vector Store Statistics ===")
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Categories: {stats.get('num_categories', 0)}")
            if stats.get('categories'):
                print(f"Available categories: {', '.join(stats['categories'])}")
        
        if postgres_searcher:
            stats = postgres_searcher.get_statistics()
            print("\n=== PostgreSQL Statistics ===")
            print(f"Total documents: {stats['overall']['total_documents']}")
            print(f"Total pages: {stats['overall']['total_pages']}")
            print(f"Total characters: {stats['overall']['total_chars']:,}")
            print("\nDocuments by category:")
            for cat in stats['by_category']:
                print(f"  {cat['category']}: {cat['count']} documents, {cat['total_pages']} pages")
        
        print()
        if postgres_searcher:
            postgres_searcher.close()
        if hybrid_searcher:
            hybrid_searcher.close()
        return
    
    if args.categories:
        categories = set()
        if hybrid_searcher:
            if hybrid_searcher.vector_searcher:
                categories.update(hybrid_searcher.vector_searcher.get_categories())
            if hybrid_searcher.postgres_searcher:
                stats = hybrid_searcher.postgres_searcher.get_statistics()
                for cat in stats['by_category']:
                    categories.add(cat['category'])
        elif vector_searcher:
            categories.update(vector_searcher.get_categories())
        if postgres_searcher:
            stats = postgres_searcher.get_statistics()
            for cat in stats['by_category']:
                categories.add(cat['category'])
        
        print("\n=== Available Categories ===")
        for cat in sorted(categories):
            print(f"  - {cat}")
        print()
        
        if postgres_searcher:
            postgres_searcher.close()
        if hybrid_searcher:
            hybrid_searcher.close()
        return
    
    # Perform search
    try:
        all_results = []
        
        if args.bates_start is not None:
            # Bates range search
            print(f"\nSearching for documents in Bates range {args.bates_start}-{args.bates_end}...")
            
            if vector_searcher:
                results = vector_searcher.search_by_bates_range(args.bates_start, args.bates_end)
                all_results.extend(results)
            
            if postgres_searcher:
                # Convert to string format for PostgreSQL
                bates_start_str = f"{args.bates_start:06d}"
                bates_end_str = f"{args.bates_end:06d}"
                
                # Search for documents in the range
                for bates_num in range(args.bates_start, args.bates_end + 1):
                    bates_str = f"{bates_num:06d}"
                    doc = postgres_searcher.get_document_by_bates(bates_str)
                    if doc and doc not in all_results:
                        all_results.append(doc)
        else:
            # Text search
            print(f"\nSearching for: '{args.query}'")
            if args.category:
                print(f"Category filter: {args.category}")
            if args.exhibit is not None:
                print(f"Exhibit filter: #{args.exhibit}")
            
            if args.search_engine == "vector":
                results = vector_searcher.search(
                    query=args.query,
                    n_results=args.num_results,
                    category=args.category,
                    exhibit_number=args.exhibit
                )
                display_results(results, verbose=args.verbose, search_type="vector")
            
            elif args.search_engine == "postgres":
                results = postgres_searcher.search_text(args.query, limit=args.num_results)
                # Filter by category if specified
                if args.category:
                    results = [r for r in results if r['category'] == args.category]
                # Filter by exhibit if specified
                if args.exhibit is not None:
                    results = [r for r in results if r['exhibit_id'] == args.exhibit]
                display_results(results, verbose=args.verbose, search_type="postgres")
            
            elif args.search_engine == "hybrid":
                # Determine search method and filters
                filters = {}
                if args.category:
                    filters['category'] = args.category
                if args.exhibit is not None:
                    filters['exhibit_number'] = args.exhibit
                
                results = hybrid_searcher.search(
                    query=args.query,
                    method=SearchMethod.HYBRID,
                    limit=args.num_results,
                    filters=filters
                )
                display_results(results, verbose=args.verbose, search_type="hybrid")
            
            elif args.search_engine == "both":
                print("\n=== Vector Search Results ===")
                if vector_searcher:
                    vector_results = vector_searcher.search(
                        query=args.query,
                        n_results=args.num_results,
                        category=args.category,
                        exhibit_number=args.exhibit
                    )
                    display_results(vector_results, verbose=args.verbose, search_type="vector")
                
                print("\n=== PostgreSQL Search Results ===")
                if postgres_searcher:
                    pg_results = postgres_searcher.search_text(args.query, limit=args.num_results)
                    # Filter by category if specified
                    if args.category:
                        pg_results = [r for r in pg_results if r['category'] == args.category]
                    # Filter by exhibit if specified
                    if args.exhibit is not None:
                        pg_results = [r for r in pg_results if r['exhibit_id'] == args.exhibit]
                    display_results(pg_results, verbose=args.verbose, search_type="postgres")
        
        if args.bates_start is not None:
            display_results(all_results, verbose=args.verbose, 
                          search_type="postgres" if postgres_searcher else "vector")
        
    except Exception as e:
        print(f"\nError: Search failed: {e}")
        sys.exit(1)
    finally:
        if postgres_searcher:
            postgres_searcher.close()
        if hybrid_searcher:
            hybrid_searcher.close()


if __name__ == "__main__":
    main()