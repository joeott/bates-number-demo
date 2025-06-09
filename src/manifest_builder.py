#!/usr/bin/env python3
"""
PDF Manifest Builder CLI

Scans directories for PDF files and creates a manifest for processing.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import scan_for_pdfs, register_pdfs_in_database, get_pending_pdfs, setup_logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def cmd_scan(args):
    """Scan directory for PDFs and create manifest."""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Directory does not exist: {input_dir}")
        return 1
    
    print(f"Scanning {input_dir} for PDF files...")
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or Path(f"manifest_{timestamp}.json")
    
    try:
        # Scan for PDFs
        manifest = scan_for_pdfs(input_dir, output_file)
        
        # Display summary
        print(f"\n{'='*60}")
        print("SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Total PDFs found: {manifest['total_files']}")
        print(f"Total size: {manifest['total_size_mb']} MB")
        
        if manifest['errors']:
            print(f"\nErrors encountered: {len(manifest['errors'])}")
            for err in manifest['errors'][:5]:  # Show first 5 errors
                print(f"  - {err['path']}: {err['error']}")
        
        # Show folder breakdown
        if manifest['summary']['by_folder']:
            print("\nPDFs by folder:")
            for folder, count in sorted(manifest['summary']['by_folder'].items()):
                print(f"  {folder}: {count} files")
        
        print(f"\nManifest saved to: {output_file}")
        
        # Register in database if requested
        if args.register:
            conn_str = os.getenv('POSTGRES_CONNECTION')
            if not conn_str:
                print("Error: POSTGRES_CONNECTION not found in environment")
                return 1
                
            print(f"\nRegistering PDFs in database...")
            path_to_uuid = register_pdfs_in_database(manifest, conn_str)
            print(f"Registered {len(path_to_uuid)} PDFs in database")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1


def cmd_status(args):
    """Show status of PDFs in manifest database."""
    conn_str = os.getenv('POSTGRES_CONNECTION')
    if not conn_str:
        print("Error: POSTGRES_CONNECTION not found in environment")
        return 1
    
    try:
        # Get pending PDFs
        pending = get_pending_pdfs(conn_str, limit=args.limit)
        
        if not pending:
            print("No pending PDFs in manifest")
            return 0
        
        print(f"\nPending PDFs: {len(pending)}")
        print(f"{'='*80}")
        print(f"{'ID':<36} {'Filename':<40} {'Size (MB)':<10}")
        print(f"{'-'*36} {'-'*40} {'-'*10}")
        
        for pdf in pending:
            size_mb = round(pdf['file_size'] / (1024 * 1024), 2) if pdf['file_size'] else 0
            print(f"{pdf['id']:<36} {pdf['file_name'][:40]:<40} {size_mb:<10}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1


def cmd_list(args):
    """List PDFs from a manifest file."""
    manifest_file = Path(args.manifest)
    
    if not manifest_file.exists():
        print(f"Error: Manifest file not found: {manifest_file}")
        return 1
    
    try:
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        print(f"\nManifest: {manifest_file}")
        print(f"Scan date: {manifest['scan_timestamp']}")
        print(f"Root directory: {manifest['root_directory']}")
        print(f"Total files: {manifest['total_files']}")
        print(f"Total size: {manifest['total_size_mb']} MB")
        
        if args.verbose:
            print(f"\n{'='*100}")
            print(f"{'Filename':<50} {'Path':<30} {'Size (MB)':<10}")
            print(f"{'-'*50} {'-'*30} {'-'*10}")
            
            for pdf in manifest['documents'][:args.limit]:
                filename = pdf['filename'][:50]
                path = pdf.get('parent_folder', 'root')[:30]
                size = pdf['size_mb']
                print(f"{filename:<50} {path:<30} {size:<10}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error reading manifest: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Manifest Builder - Scan and manage PDF files for processing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory for PDFs')
    scan_parser.add_argument('input_dir', help='Directory to scan')
    scan_parser.add_argument('-o', '--output', help='Output manifest file')
    scan_parser.add_argument('-r', '--register', action='store_true', 
                           help='Register PDFs in database')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show manifest database status')
    status_parser.add_argument('-l', '--limit', type=int, default=20,
                             help='Limit number of results shown')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List PDFs from manifest file')
    list_parser.add_argument('manifest', help='Manifest JSON file')
    list_parser.add_argument('-v', '--verbose', action='store_true',
                           help='Show detailed file list')
    list_parser.add_argument('-l', '--limit', type=int, default=50,
                           help='Limit number of files shown')
    
    # Add debug flag
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Execute command
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'list':
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())