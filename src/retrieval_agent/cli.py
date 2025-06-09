# File: src/retrieval_agent/cli.py
"""
Command-line interface for the Iterative Retrieval Agent.

This CLI provides a simple way to test and interact with the retrieval agent
using production data and configurations.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure src directory is in Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval_agent.main_retriever import IterativeRetrieverAgent
from src.retrieval_agent import agent_config
from src.utils import setup_logging
# Config already loads env vars on import
import src.config

logger = logging.getLogger(__name__)

def setup_langsmith():
    """Configure LangSmith tracing if environment variables are set."""
    # Check if LangSmith is configured
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project = os.getenv("LANGCHAIN_PROJECT", "BatesNumbering-RetrievalAgent")
        
        if api_key:
            logger.info(f"LangSmith tracing enabled for project: {project}")
            # Ensure the project name is set
            os.environ["LANGCHAIN_PROJECT"] = project
        else:
            logger.warning("LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY is not set")
    else:
        logger.info("LangSmith tracing is not enabled (set LANGCHAIN_TRACING_V2=true to enable)")

def format_output(query: str, answer: str, execution_time: float):
    """Format the output for display."""
    separator = "=" * 80
    
    output = f"""
{separator}
QUERY: {query}
{separator}

ANSWER:
{answer}

{separator}
Execution Time: {execution_time:.2f} seconds
{separator}
"""
    return output

def save_result(query: str, answer: str, output_dir: Path):
    """Save the query and answer to a file for future reference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retrieval_result_{timestamp}.txt"
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"\nAnswer:\n{answer}\n")
    
    return filepath

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iterative Retrieval Agent CLI - Advanced legal document search and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple query
  python -m src.retrieval_agent.cli "What is the contract price?"
  
  # Complex multi-aspect query
  python -m src.retrieval_agent.cli "Find all evidence of breach of contract by Pal's Glass"
  
  # With custom iterations
  python -m src.retrieval_agent.cli --max-iterations 5 "Identify negligence elements in the case"
  
  # Save results to file
  python -m src.retrieval_agent.cli --save-results "What damages were claimed?"
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="The legal query to process"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=f"Maximum number of retrieval iterations (default: {agent_config.MAX_ITERATIONS})"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the results to a file in the output directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/retrieval_results"),
        help="Directory to save results (default: output/retrieval_results)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Disable LangSmith tracing even if configured"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    # Setup LangSmith unless explicitly disabled
    if not args.no_langsmith:
        setup_langsmith()
    
    # Log the query
    logger.info(f"Processing query: '{args.query}'")
    
    # Initialize the agent
    try:
        logger.info("Initializing IterativeRetrieverAgent...")
        if args.max_iterations:
            agent = IterativeRetrieverAgent(max_iterations=args.max_iterations)
        else:
            agent = IterativeRetrieverAgent()  # Uses default from agent_config
        
        logger.info("Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"\nERROR: Failed to initialize the retrieval agent: {e}")
        print("Please check your configuration and ensure the vector store is populated.")
        return 1
    
    # Create run configuration for LangSmith
    run_config = {
        "metadata": {
            "user_query": args.query,
            "cli_invocation": True,
            "max_iterations": args.max_iterations or agent_config.MAX_ITERATIONS
        },
        "tags": ["cli", "iterative_retrieval"],
        "run_name": f"CLI-{args.query[:50].replace(' ', '_')}"
    }
    
    # Execute the query
    try:
        import time
        start_time = time.time()
        
        logger.info("Invoking agent with query...")
        answer = agent.invoke(args.query, run_config=run_config)
        
        execution_time = time.time() - start_time
        logger.info(f"Query completed in {execution_time:.2f} seconds")
        
        # Format and display the output
        output = format_output(args.query, answer, execution_time)
        print(output)
        
        # Save results if requested
        if args.save_results:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_result(args.query, answer, args.output_dir)
            print(f"\nResults saved to: {filepath}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Query interrupted by user")
        print("\n\nQuery interrupted.")
        return 130
        
    except Exception as e:
        logger.error(f"An error occurred during agent invocation: {e}", exc_info=True)
        print(f"\nERROR: An unexpected error occurred: {e}")
        print("\nPlease check the logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())