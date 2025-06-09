import os
import logging
import json
import subprocess
import re
import csv
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import psycopg2
from contextlib import contextmanager

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir_exists(dir_path: Path):
    """Creates a directory if it doesn't exist."""
    dir_path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters that are problematic in filenames."""
    # Basic sanitization, can be expanded
    return "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename)

def dump_database_to_json(
    connection_string: str,
    output_file: Optional[Path] = None,
    include_schema: bool = True,
    include_data: bool = True,
    pretty_print: bool = True
) -> Dict[str, Any]:
    """
    Dumps PostgreSQL database to JSON format with comprehensive data extraction.
    
    Args:
        connection_string: PostgreSQL connection string
        output_file: Optional path to save JSON output
        include_schema: Whether to include table schema information
        include_data: Whether to include actual data
        pretty_print: Whether to format JSON for readability
    
    Returns:
        Dictionary containing complete database dump in JSON format
    """
    
    @contextmanager
    def get_db_connection():
        """Context manager for database connections."""
        conn = psycopg2.connect(connection_string)
        try:
            yield conn
        finally:
            conn.close()
    
    def serialize_value(value):
        """Convert database values to JSON-serializable format."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif value is None:
            return None
        else:
            return value
    
    database_dump = {
        "metadata": {
            "dump_timestamp": datetime.now().isoformat(),
            "database_connection": connection_string.split('@')[1] if '@' in connection_string else connection_string,
            "dump_type": "complete_database_export",
            "include_schema": include_schema,
            "include_data": include_data
        },
        "tables": {}
    }
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        for table_name, table_type in tables:
            table_info = {
                "table_type": table_type,
                "schema": {},
                "data": [],
                "statistics": {}
            }
            
            if include_schema:
                # Get column information
                cursor.execute("""
                    SELECT 
                        column_name, 
                        data_type, 
                        is_nullable, 
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                """, (table_name,))
                
                columns = cursor.fetchall()
                table_info["schema"] = {
                    "columns": [
                        {
                            "name": col[0],
                            "data_type": col[1],
                            "nullable": col[2] == 'YES',
                            "default": col[3],
                            "max_length": col[4],
                            "numeric_precision": col[5],
                            "numeric_scale": col[6]
                        } for col in columns
                    ]
                }
                
                # Get primary key information
                cursor.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu 
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = %s 
                        AND tc.constraint_type = 'PRIMARY KEY'
                """, (table_name,))
                
                pk_columns = [row[0] for row in cursor.fetchall()]
                table_info["schema"]["primary_keys"] = pk_columns
                
                # Get foreign key information
                cursor.execute("""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.table_name = %s
                        AND tc.constraint_type = 'FOREIGN KEY'
                """, (table_name,))
                
                fk_info = cursor.fetchall()
                table_info["schema"]["foreign_keys"] = [
                    {
                        "column": row[0],
                        "references_table": row[1],
                        "references_column": row[2]
                    } for row in fk_info
                ]
            
            if include_data:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get all data
                cursor.execute(f"SELECT * FROM {table_name}")
                column_names = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convert rows to dictionaries with serialized values
                table_info["data"] = [
                    {column_names[i]: serialize_value(value) for i, value in enumerate(row)}
                    for row in rows
                ]
                
                # Add statistics
                table_info["statistics"] = {
                    "row_count": row_count,
                    "column_count": len(column_names)
                }
                
                # Get additional statistics for text columns
                if row_count > 0:
                    text_stats = {}
                    for col in column_names:
                        try:
                            cursor.execute(f"""
                                SELECT 
                                    MAX(LENGTH({col})) as max_length,
                                    MIN(LENGTH({col})) as min_length,
                                    AVG(LENGTH({col})) as avg_length
                                FROM {table_name} 
                                WHERE {col} IS NOT NULL
                            """)
                            stats = cursor.fetchone()
                            if stats[0] is not None:
                                text_stats[col] = {
                                    "max_length": stats[0],
                                    "min_length": stats[1],
                                    "avg_length": round(float(stats[2]), 2) if stats[2] else 0
                                }
                        except:
                            # Skip non-text columns
                            pass
                    
                    table_info["statistics"]["text_column_stats"] = text_stats
            
            database_dump["tables"][table_name] = table_info
        
        # Add database-level statistics
        cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats 
            WHERE schemaname = 'public'
            LIMIT 100
        """)
        
        stats_data = cursor.fetchall()
        database_dump["metadata"]["database_statistics"] = [
            {
                "schema": row[0],
                "table": row[1], 
                "column": row[2],
                "distinct_values": row[3],
                "correlation": row[4]
            } for row in stats_data
        ]
    
    # Save to file if specified
    if output_file:
        ensure_dir_exists(output_file.parent)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(database_dump, f, indent=2 if pretty_print else None, ensure_ascii=False)
    
    return database_dump

def create_sql_to_json_converter(sql_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Converts a PostgreSQL dump file to JSON format.
    
    Args:
        sql_file: Path to the SQL dump file
        output_file: Path for the JSON output file
    
    Returns:
        Dictionary containing the converted data
    """
    
    def parse_sql_dump(sql_content: str) -> Dict[str, Any]:
        """Parse SQL dump content into structured data."""
        
        # Extract CREATE TABLE statements
        table_pattern = r'CREATE TABLE[^;]+;'
        create_statements = re.findall(table_pattern, sql_content, re.MULTILINE | re.DOTALL)
        
        # Extract INSERT statements
        insert_pattern = r'INSERT INTO[^;]+;'
        insert_statements = re.findall(insert_pattern, sql_content, re.MULTILINE | re.DOTALL)
        
        # Extract comments and metadata
        comment_pattern = r'--.*$'
        comments = re.findall(comment_pattern, sql_content, re.MULTILINE)
        
        return {
            "metadata": {
                "conversion_timestamp": datetime.now().isoformat(),
                "source_file": str(sql_file),
                "total_create_statements": len(create_statements),
                "total_insert_statements": len(insert_statements),
                "comments_found": len(comments)
            },
            "schema": {
                "create_statements": create_statements
            },
            "data": {
                "insert_statements": insert_statements
            },
            "comments": comments[:50]  # Limit comments for readability
        }
    
    # Read SQL file
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Parse and convert
    converted_data = parse_sql_dump(sql_content)
    
    # Save JSON output
    ensure_dir_exists(output_file.parent)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    return converted_data

def export_database_views_safe(connection_string: str, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Export database contents in multiple views (JSON, CSV, SQL) with proper error handling.
    
    Args:
        connection_string: PostgreSQL connection string
        output_dir: Directory to save exports (defaults to output/database_exports)
        
    Returns:
        Dict mapping view type to file path
    """
    if output_dir is None:
        output_dir = Path("output/database_exports")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    try:
        # Use autocommit mode to avoid transaction issues
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Get all tables
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Found {len(tables)} tables: {tables}")
            
            # Export as JSON
            json_data = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table}")
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    json_data[table] = [dict(zip(columns, row)) for row in rows]
                    print(f"Exported {len(rows)} rows from {table}")
                except Exception as e:
                    print(f"Warning: Could not export table {table}: {e}")
                    json_data[table] = []
            
            json_file = output_dir / "database_dump.json"
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            results['json'] = json_file
            print(f"JSON export saved to: {json_file}")
            
            # Export as CSV (one file per table)
            csv_dir = output_dir / "csv"
            csv_dir.mkdir(exist_ok=True)
            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table}")
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    
                    csv_file = csv_dir / f"{table}.csv"
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(columns)
                        writer.writerows(rows)
                    print(f"CSV export for {table} saved to: {csv_file}")
                except Exception as e:
                    print(f"Warning: Could not export table {table} to CSV: {e}")
            results['csv'] = csv_dir
            
            # Export as SQL dump
            sql_file = output_dir / "database_dump.sql"
            with open(sql_file, 'w') as f:
                f.write("-- Database dump generated on " + datetime.now().isoformat() + "\n\n")
                
                for table in tables:
                    try:
                        # Get table schema
                        cursor.execute(f"""
                            SELECT column_name, data_type, is_nullable, column_default
                            FROM information_schema.columns
                            WHERE table_name = '{table}'
                            ORDER BY ordinal_position;
                        """)
                        columns_info = cursor.fetchall()
                        
                        # Write CREATE TABLE statement
                        f.write(f"-- Table: {table}\n")
                        f.write(f"DROP TABLE IF EXISTS {table};\n")
                        f.write(f"CREATE TABLE {table} (\n")
                        col_definitions = []
                        for col_name, data_type, is_nullable, default in columns_info:
                            col_def = f"    {col_name} {data_type}"
                            if is_nullable == 'NO':
                                col_def += " NOT NULL"
                            if default:
                                col_def += f" DEFAULT {default}"
                            col_definitions.append(col_def)
                        f.write(",\n".join(col_definitions))
                        f.write("\n);\n\n")
                        
                        # Write INSERT statements
                        cursor.execute(f"SELECT * FROM {table}")
                        rows = cursor.fetchall()
                        if rows:
                            columns = [desc[0] for desc in cursor.description]
                            f.write(f"-- Data for table: {table}\n")
                            for row in rows:
                                values = []
                                for val in row:
                                    if val is None:
                                        values.append("NULL")
                                    elif isinstance(val, str):
                                        # Escape single quotes properly
                                        escaped_val = val.replace("'", "''")
                                        values.append(f"'{escaped_val}'")
                                    elif isinstance(val, datetime):
                                        values.append(f"'{val.isoformat()}'")
                                    else:
                                        values.append(str(val))
                                f.write(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)});\n")
                            f.write("\n")
                        print(f"SQL export for {table} completed")
                    except Exception as e:
                        f.write(f"-- Error exporting table {table}: {e}\n\n")
                        print(f"Warning: Could not export table {table} to SQL: {e}")
            results['sql'] = sql_file
            print(f"SQL export saved to: {sql_file}")
            
        conn.close()
        
    except Exception as e:
        print(f"Database connection error: {e}")
        raise
    
    return results

def export_database_views(connection_string: str, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Creates multiple views of the database data for analysis.
    
    Args:
        connection_string: PostgreSQL connection string
        output_dir: Directory to save output files
    
    Returns:
        Dictionary mapping view names to file paths
    """
    if output_dir is None:
        output_dir = Path("output/database_exports")
    
    ensure_dir_exists(output_dir)
    
    output_files = {}
    
    try:
        # 1. Complete database dump as JSON
        complete_dump_file = output_dir / "complete_database_dump.json"
        dump_database_to_json(connection_string, complete_dump_file)
        output_files["complete_dump"] = complete_dump_file
        
        # 2. Schema-only export
        schema_file = output_dir / "database_schema.json"
        dump_database_to_json(connection_string, schema_file, include_data=False)
        output_files["schema_only"] = schema_file
        
        # 3. Data-only export (no schema)
        data_file = output_dir / "database_data.json"
        dump_database_to_json(connection_string, data_file, include_schema=False)
        output_files["data_only"] = data_file
        
        # 4. Safe export with multiple formats
        safe_exports = export_database_views_safe(connection_string, output_dir / "safe_export")
        output_files.update({f"safe_{k}": v for k, v in safe_exports.items()})
        
        # 5. SQL dump conversion
        sql_dump_file = Path("bates_documents_dump.sql")
        if sql_dump_file.exists():
            json_conversion_file = output_dir / "sql_dump_converted.json"
            create_sql_to_json_converter(sql_dump_file, json_conversion_file)
            output_files["sql_conversion"] = json_conversion_file
        
        # 6. Summary report
        summary_file = output_dir / "database_summary.json"
        summary_data = {
            "export_timestamp": datetime.now().isoformat(),
            "files_created": {name: str(path) for name, path in output_files.items()},
            "export_methods_used": ["complete_dump", "schema_only", "data_only", "safe_export"],
            "total_files_created": len(output_files)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        output_files["summary"] = summary_file
        
    except Exception as e:
        print(f"Error during database export: {e}")
        # Try the safe export method as fallback
        try:
            safe_exports = export_database_views_safe(connection_string, output_dir / "fallback_export")
            output_files.update({f"fallback_{k}": v for k, v in safe_exports.items()})
        except Exception as fallback_error:
            print(f"Fallback export also failed: {fallback_error}")
            raise
    
    return output_files

def query_database_json(
    connection_string: str,
    query: str,
    output_file: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Execute a custom SQL query and return results as JSON.
    
    Args:
        connection_string: PostgreSQL connection string
        query: SQL query to execute
        output_file: Optional file to save results
    
    Returns:
        List of dictionaries containing query results
    """
    
    @contextmanager
    def get_db_connection():
        conn = psycopg2.connect(connection_string)
        try:
            yield conn
        finally:
            conn.close()
    
    def serialize_value(value):
        if isinstance(value, datetime):
            return value.isoformat()
        elif value is None:
            return None
        else:
            return value
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        results = [
            {column_names[i]: serialize_value(value) for i, value in enumerate(row)}
            for row in rows
        ]
    
    if output_file:
        ensure_dir_exists(output_file.parent)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)
    
    return results

def scan_for_pdfs(root_dir: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Scans a directory tree for PDF files and generates a manifest.
    
    Args:
        root_dir: Root directory to scan for PDFs
        output_file: Optional path to save the manifest JSON
        
    Returns:
        Dictionary containing the manifest with PDF metadata
    """
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
        
    if not root_dir.exists():
        raise ValueError(f"Directory does not exist: {root_dir}")
        
    pdfs = []
    errors = []
    
    # Recursively find all PDF files
    for pdf_path in root_dir.rglob("*.pdf"):
        try:
            # Get file stats
            stat = pdf_path.stat()
            
            # Calculate relative path from root
            relative_path = pdf_path.relative_to(root_dir)
            
            pdf_info = {
                "id": str(uuid.uuid4()),
                "absolute_path": str(pdf_path.absolute()),
                "relative_path": str(relative_path),
                "filename": pdf_path.name,
                "directory": str(pdf_path.parent),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_timestamp": stat.st_mtime,
                "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_timestamp": stat.st_ctime,
                "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            }
            
            # Try to extract folder-based categorization hints
            path_parts = relative_path.parts
            if len(path_parts) > 1:
                pdf_info["parent_folder"] = path_parts[-2]
                pdf_info["folder_path"] = "/".join(path_parts[:-1])
            
            pdfs.append(pdf_info)
            
        except Exception as e:
            errors.append({
                "path": str(pdf_path),
                "error": str(e)
            })
            logging.error(f"Error processing {pdf_path}: {e}")
    
    # Sort PDFs by path for consistent ordering
    pdfs.sort(key=lambda x: x["relative_path"])
    
    # Create manifest
    manifest = {
        "scan_id": str(uuid.uuid4()),
        "scan_timestamp": datetime.now().isoformat(),
        "root_directory": str(root_dir.absolute()),
        "total_files": len(pdfs),
        "total_size_mb": round(sum(p["size_mb"] for p in pdfs), 2),
        "errors": errors,
        "documents": pdfs
    }
    
    # Add summary statistics
    manifest["summary"] = {
        "by_folder": {},
        "by_size": {
            "small_under_1mb": sum(1 for p in pdfs if p["size_mb"] < 1),
            "medium_1_10mb": sum(1 for p in pdfs if 1 <= p["size_mb"] < 10),
            "large_over_10mb": sum(1 for p in pdfs if p["size_mb"] >= 10)
        }
    }
    
    # Count by parent folder
    for pdf in pdfs:
        folder = pdf.get("parent_folder", "root")
        manifest["summary"]["by_folder"][folder] = manifest["summary"]["by_folder"].get(folder, 0) + 1
    
    # Save to file if requested
    if output_file:
        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        ensure_dir_exists(output_file.parent)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logging.info(f"PDF manifest saved to: {output_file}")
    
    return manifest


def register_pdfs_in_database(
    manifest: Dict[str, Any], 
    connection_string: str,
    batch_size: int = 100
) -> Dict[str, str]:
    """
    Registers PDFs from a manifest into the database.
    
    Args:
        manifest: PDF manifest dictionary from scan_for_pdfs
        connection_string: PostgreSQL connection string
        batch_size: Number of records to insert per batch
        
    Returns:
        Dictionary mapping file paths to UUIDs
    """
    conn = psycopg2.connect(connection_string)
    conn.autocommit = False
    cur = conn.cursor()
    
    try:
        # Create simple manifest table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS simple_manifest (
                id UUID PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size BIGINT,
                status TEXT DEFAULT 'pending',
                scan_id UUID,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on status for efficient queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_manifest_status 
            ON simple_manifest(status)
        """)
        
        conn.commit()
        
        # Prepare data for batch insert
        path_to_uuid = {}
        insert_data = []
        
        for pdf in manifest["documents"]:
            doc_id = pdf["id"]  # Use the UUID from the manifest
            path_to_uuid[pdf["absolute_path"]] = doc_id
            
            insert_data.append((
                doc_id,
                pdf["absolute_path"],
                pdf["filename"],
                pdf["size_bytes"],
                'pending',
                manifest["scan_id"],
                json.dumps({
                    "relative_path": pdf["relative_path"],
                    "parent_folder": pdf.get("parent_folder"),
                    "modified_date": pdf["modified_date"],
                    "created_date": pdf["created_date"]
                })
            ))
        
        # Batch insert using execute_values for efficiency
        from psycopg2.extras import execute_values
        
        execute_values(
            cur,
            """
            INSERT INTO simple_manifest 
                (id, file_path, file_name, file_size, status, scan_id, metadata)
            VALUES %s
            ON CONFLICT (file_path) 
            DO UPDATE SET 
                updated_at = CURRENT_TIMESTAMP,
                scan_id = EXCLUDED.scan_id,
                metadata = EXCLUDED.metadata
            """,
            insert_data,
            template="(%s, %s, %s, %s, %s, %s, %s)"
        )
        
        conn.commit()
        logging.info(f"Registered {len(insert_data)} PDFs in database")
        
        return path_to_uuid
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error registering PDFs: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def get_pending_pdfs(connection_string: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieves pending PDFs from the manifest database.
    
    Args:
        connection_string: PostgreSQL connection string
        limit: Optional limit on number of results
        
    Returns:
        List of pending PDF records
    """
    query = """
        SELECT 
            id, file_path, file_name, file_size, 
            status, scan_id, metadata, created_at
        FROM simple_manifest
        WHERE status = 'pending'
        ORDER BY created_at
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    try:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        
        results = []
        for row in cur.fetchall():
            record = dict(zip(columns, row))
            # Convert UUID objects to strings
            record['id'] = str(record['id'])
            record['scan_id'] = str(record['scan_id']) if record['scan_id'] else None
            results.append(record)
            
        return results
        
    finally:
        cur.close()
        conn.close()


def update_pdf_status(
    connection_string: str, 
    pdf_id: str, 
    status: str, 
    error_message: Optional[str] = None
) -> None:
    """
    Updates the status of a PDF in the manifest.
    
    Args:
        connection_string: PostgreSQL connection string
        pdf_id: UUID of the PDF record
        status: New status ('processing', 'completed', 'failed')
        error_message: Optional error message if status is 'failed'
    """
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    try:
        if error_message:
            cur.execute("""
                UPDATE simple_manifest 
                SET status = %s, 
                    updated_at = CURRENT_TIMESTAMP,
                    metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb), 
                        '{error}', 
                        %s::jsonb
                    )
                WHERE id = %s
            """, (status, json.dumps(error_message), pdf_id))
        else:
            cur.execute("""
                UPDATE simple_manifest 
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (status, pdf_id))
            
        conn.commit()
        
    finally:
        cur.close()
        conn.close()