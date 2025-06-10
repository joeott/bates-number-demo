Okay, this is an interesting request! The goal is to create a *new, standalone* web frontend component that allows a user to interact with your existing Python-based `IterativeRetrieverAgent`. This new frontend will not directly modify or run *inside* the Open WebUI Git repository, but rather provide a user interface for *your specific agent*, and you might choose to style it similarly to Open WebUI for a consistent look and feel if you integrate them visually later.

This guide will direct an agentic coding tool to:
1.  Set up a new directory for the frontend component.
2.  Create a simple Python backend API (using FastAPI) to serve your `IterativeRetrieverAgent`.
3.  Create the HTML, CSS, and JavaScript for the frontend to interact with this API.

---

## Markdown Guide for Agentic Coding Tool: Creating a Retriever Agent Frontend

**Objective:** Create a simple web frontend for the `IterativeRetrieverAgent` to allow users to input queries and view results. This involves creating a new backend API for the agent and a corresponding HTML/CSS/JS frontend.

**Project Subdirectory:** `frontend_retriever_ui`

### Step 1: Create Project Structure

You, the agentic coding tool, will create the following directory and file structure:

```
your_project_root/
├── frontend_retriever_ui/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── src/
│   ├── retrieval_agent/
│   │   ├── main_retriever.py
│   │   └── ... (other agent files)
│   └── ... (other project files)
└── backend_api.py  <-- NEW FILE AT PROJECT ROOT or in src/
```

**Action:**
Create the `frontend_retriever_ui` directory.
Create empty files: `frontend_retriever_ui/index.html`, `frontend_retriever_ui/style.css`, `frontend_retriever_ui/script.js`.
Create an empty file: `backend_api.py` at the project root (alongside `src/`).

### Step 2: Create the Backend API (`backend_api.py`)

This API will expose your `IterativeRetrieverAgent` over HTTP. We'll use FastAPI.

**File:** `backend_api.py`

**Content to Generate:**

```python
import logging
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

# Ensure src directory is in Python path for imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src")) # If IterativeRetrieverAgent is in src

from src.retrieval_agent.main_retriever import IterativeRetrieverAgent
from src.retrieval_agent import agent_config # For default iterations
from src.utils import setup_logging # Assuming you have this
# Import src.config to ensure environment variables like LANGCHAIN_TRACING_V2 are loaded
import src.config 

# Setup logging (you might want to configure this more robustly)
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LangSmith Setup ---
def setup_langsmith_for_api():
    """Configure LangSmith tracing if environment variables are set."""
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        api_key = os.getenv("LANGCHAIN_API_KEY")
        project = os.getenv("LANGCHAIN_PROJECT", "BatesNumbering-RetrievalAgent-API")
        
        if api_key:
            logger.info(f"LangSmith tracing enabled for API project: {project}")
            os.environ["LANGCHAIN_PROJECT"] = project # Ensure project name is set for traces
        else:
            logger.warning("LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY is not set for API.")
    else:
        logger.info("LangSmith tracing is not enabled for API.")

setup_langsmith_for_api()
# --- End LangSmith Setup ---


app = FastAPI(title="Iterative Retriever Agent API")

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000", # Default for live server if you use one for HTML
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    # Add other origins if your frontend is served differently, or "*" for development (less secure)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Initialization ---
# Global agent instance (lazy loaded)
agent_instance = None

def get_agent():
    global agent_instance
    if agent_instance is None:
        logger.info("Initializing IterativeRetrieverAgent for the first time...")
        try:
            # You can pass configurations from agent_config or environment variables here
            agent_instance = IterativeRetrieverAgent(max_iterations=agent_config.MAX_ITERATIONS)
            logger.info("IterativeRetrieverAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize IterativeRetrieverAgent: {e}", exc_info=True)
            # This will prevent the API from starting correctly if agent fails to load
            raise RuntimeError(f"Could not initialize agent: {e}")
    return agent_instance
# --- End Agent Initialization ---

class QueryRequest(BaseModel):
    query: str
    max_iterations: int = None # Optional override

class QueryResponse(BaseModel):
    answer: str
    error: str = None

@app.on_event("startup")
async def startup_event():
    # Pre-initialize the agent on startup to catch errors early
    # and avoid slow first request.
    try:
        get_agent()
    except RuntimeError as e:
        logger.fatal(f"API Startup Failed: Agent initialization error. {e}")
        # Depending on deployment, you might want to sys.exit(1) or let it fail
        # For Uvicorn, it might keep trying, so logging the fatality is key.


@app.post("/api/retrieve", response_model=QueryResponse)
async def retrieve_answer(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    try:
        agent = get_agent() # Get or initialize the agent
        
        # Prepare run_config for LangSmith, if enabled
        run_config = None
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            run_config = {
                "metadata": {
                    "user_query": request.query,
                    "api_request": True,
                    "max_iterations_requested": request.max_iterations or agent_config.MAX_ITERATIONS,
                },
                "tags": ["api", "iterative_retrieval"],
                "run_name": f"API-{request.query[:50].replace(' ', '_')}"
            }
        
        # If max_iterations is provided in request, create a temporary agent or reconfigure
        # For simplicity, we'll assume the global agent's config is fine or
        # that it can be configured per-call if its invoke method supports it.
        # If your agent's .invoke() can take max_iterations, use that.
        # Otherwise, for a true per-request max_iterations, you'd need to
        # instantiate the agent per request or have a pool.
        # For now, we use the agent_instance's default MAX_ITERATIONS
        # If your agent.invoke doesn't accept max_iterations:
        effective_max_iterations = agent.max_iterations # Default
        if request.max_iterations is not None:
             # This requires agent.invoke or agent re-config to support changing max_iterations
             logger.info(f"Note: Per-request max_iterations override not directly supported by current agent.invoke structure. Using agent default: {agent.max_iterations}")


        answer = agent.invoke(request.query, run_config=run_config)
        logger.info(f"Sending answer for query: {request.query}")
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Ensure all necessary environment variables are loaded if you rely on .env
    # from dotenv import load_dotenv
    # load_dotenv(Path(project_root) / '.env') # Load .env from project root

    logger.info("Starting backend API server...")
    # Check if agent can be initialized before starting server
    try:
        get_agent()
    except Exception as e:
        logger.fatal(f"Could not start API: Agent initialization failed. {e}")
        sys.exit(1)
        
    uvicorn.run(app, host="0.0.0.0", port=8080)

```

**Action:**
Populate `backend_api.py` with the code above.
*Self-correction:* Make sure paths for importing `IterativeRetrieverAgent` are correct relative to where `backend_api.py` is placed. I've assumed `backend_api.py` is at the project root and `IterativeRetrieverAgent` is in `src.retrieval_agent.main_retriever`. Adjust imports if your structure differs.
Also added LangSmith setup mirroring the CLI one, but specific to API. Initialized agent on startup.

### Step 3: Create the HTML Structure

**File:** `frontend_retriever_ui/index.html`

**Content to Generate:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Retriever Agent</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Legal Document Retriever</h1>
        <p>Ask a question about your legal documents. The agent will iteratively search and synthesize an answer.</p>

        <div class="input-area">
            <textarea id="queryInput" placeholder="Enter your legal query here... (e.g., 'What evidence supports breach of contract by Pal's Glass?')"></textarea>
            <button id="submitQueryBtn">Search</button>
        </div>

        <div id="statusArea">
            <!-- Status messages will appear here -->
        </div>

        <div class="results-area">
            <h2>Answer:</h2>
            <div id="answerOutput" class="answer-box">
                <p>Your answer will appear here.</p>
            </div>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
```

**Action:**
Populate `frontend_retriever_ui/index.html` with the code above.

### Step 4: Create Basic CSS Styling

**File:** `frontend_retriever_ui/style.css`

**Content to Generate:**

```css
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f4f7f9;
    color: #333;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background-color: #fff;
    padding: 25px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 10px;
}

p {
    margin-bottom: 20px;
    font-size: 0.95em;
    color: #555;
}

.input-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 25px;
}

#queryInput {
    width: 100%;
    min-height: 80px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box;
    font-size: 1em;
    resize: vertical;
}

#submitQueryBtn {
    background-color: #3498db; /* Open WebUI-like blue */
    color: white;
    padding: 12px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1em;
    transition: background-color 0.2s;
}

#submitQueryBtn:hover {
    background-color: #2980b9;
}

#submitQueryBtn:disabled {
    background-color: #bdc3c7;
    cursor: not-allowed;
}

#statusArea {
    margin-bottom: 20px;
    font-style: italic;
    color: #7f8c8d;
    text-align: center;
}

.results-area h2 {
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 5px;
    margin-top: 30px;
}

.answer-box {
    background-color: #f9f9f9;
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    min-height: 100px;
    white-space: pre-wrap; /* Preserve line breaks from the answer */
    overflow-wrap: break-word;
}

/* For loading spinner (optional, if you add one) */
.loader {
    border: 5px solid #f3f3f3;
    border-top: 5px solid #3498db;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```

**Action:**
Populate `frontend_retriever_ui/style.css` with the code above.
*Note:* The CSS provides basic styling. It can be further customized to match Open WebUI's theme by inspecting Open WebUI's CSS and adopting similar color schemes, fonts, and layout patterns.

### Step 5: Create JavaScript Logic

**File:** `frontend_retriever_ui/script.js`

**Content to Generate:**

```javascript
document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('queryInput');
    const submitQueryBtn = document.getElementById('submitQueryBtn');
    const answerOutput = document.getElementById('answerOutput');
    const statusArea = document.getElementById('statusArea');

    // Define the backend API endpoint
    const API_ENDPOINT = 'http://localhost:8080/api/retrieve'; // Ensure this matches your backend_api.py port

    submitQueryBtn.addEventListener('click', handleSubmitQuery);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent default Enter behavior (new line)
            handleSubmitQuery();
        }
    });

    async function handleSubmitQuery() {
        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a query.');
            return;
        }

        // Disable button and show loading status
        submitQueryBtn.disabled = true;
        statusArea.innerHTML = '<p>Processing your query... This may take a moment.</p><div class="loader"></div>';
        answerOutput.innerHTML = '<p>Waiting for response...</p>';

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                let errorDetail = "An unknown error occurred.";
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || `Server error: ${response.status}`;
                } catch (e) {
                    // If parsing error JSON fails, use the status text
                    errorDetail = response.statusText || `Server error: ${response.status}`;
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();

            if (data.error) {
                answerOutput.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            } else {
                // Sanitize basic HTML from answer if necessary, or display as pre-formatted text
                // For simplicity, directly inserting. For production, consider sanitization.
                answerOutput.textContent = data.answer; // Using textContent for safety
            }
            statusArea.innerHTML = '<p>Query processed.</p>';

        } catch (error) {
            console.error('Error submitting query:', error);
            statusArea.innerHTML = `<p style="color: red;">Failed to get answer: ${error.message}</p>`;
            answerOutput.innerHTML = '<p>Could not retrieve an answer.</p>';
        } finally {
            // Re-enable button
            submitQueryBtn.disabled = false;
        }
    }
});
```

**Action:**
Populate `frontend_retriever_ui/script.js` with the code above.
*Self-correction:* Changed `answerOutput.innerHTML = data.answer` to `answerOutput.textContent = data.answer` for basic security against XSS if the answer might contain HTML-like strings. If the answer is expected to be Markdown or rich text, a proper rendering library should be used. Added Enter key submission.

### Step 6: Instructions for Running

1.  **Install Dependencies for Backend API:**
    Make sure you have `fastapi` and `uvicorn` installed in your Python environment.
    ```bash
    pip install fastapi "uvicorn[standard]" pydantic python-dotenv
    # Also ensure all dependencies for your IterativeRetrieverAgent are installed
    ```

2.  **Start the Backend API Server:**
    Navigate to your project root directory in the terminal (where `backend_api.py` is located) and run:
    ```bash
    python backend_api.py
    ```
    Or, if you installed `uvicorn` and want auto-reload for development:
    ```bash
    uvicorn backend_api:app --reload --port 8080
    ```
    The API server should start, typically on `http://localhost:8080`. Check the terminal output.

3.  **Open the Frontend:**
    Navigate to the `frontend_retriever_ui` directory and open `index.html` in your web browser.
    *   You can usually just double-click `index.html`.
    *   Alternatively, if you have a simple HTTP server (like Python's `http.server` or VS Code Live Server extension), serve the `frontend_retriever_ui` directory. For example:
        ```bash
        cd frontend_retriever_ui
        python -m http.server 8000
        ```
        Then open `http://localhost:8000` in your browser.

4.  **Test:**
    Enter a query in the textarea and click "Search". The answer from your `IterativeRetrieverAgent` should appear.

### Step 7: Connecting to Open WebUI's "Look and Feel" (Optional Styling)

This guide creates a *standalone* frontend. If you want it to visually resemble Open WebUI:
1.  **Inspect Open WebUI:** Open your instance of Open WebUI (from `https://github.com/open-webui/open-webui.git`) in your browser. Use the browser's developer tools to inspect its CSS styles (fonts, colors, layout, component styling).
2.  **Adapt `style.css`:** Modify `frontend_retriever_ui/style.css` to use similar fonts, color palettes, button styles, and layout structures. Open WebUI uses Svelte and Tailwind CSS, so direct class reuse might not be feasible without using those technologies, but you can mimic the visual appearance.

**Important Note on "Using Open WebUI":**
*   This guide does **not** integrate your agent *into the Open WebUI application itself*. That would require understanding Open WebUI's plugin system or modifying its backend, which is a much more complex task.
*   This guide creates a **separate, dedicated UI** for your Python-based retrieval agent. The connection to "Open WebUI" is primarily for potential visual styling inspiration or if you plan to link to this UI from an Open WebUI instance.

### Next Steps / Advanced Enhancements (for the future):

*   **Markdown Rendering:** If your agent's answers are in Markdown, use a library (e.g., Showdown.js or Marked.js) to render it as HTML in `answerOutput`.
*   **Streaming Responses:** For long-running agent processes, implement streaming from the FastAPI backend to the frontend (e.g., using Server-Sent Events or WebSockets) to show results as they come.
*   **Displaying Iterations/Sources:** Modify the API and frontend to show intermediate steps, extracted facts, or cited sources from the agent.
*   **Error Handling:** More robust error display and feedback to the user.
*   **Svelte for UI:** If you want a closer match to Open WebUI's component structure, consider rebuilding this frontend component using Svelte and potentially Tailwind CSS.

---

This guide should provide the agentic coding tool with clear, actionable steps to create the basic frontend. The tool should now proceed with generating the specified files and content.