# Legal Document Retriever UI

A web-based user interface for the Iterative Retrieval Agent, providing an intuitive way to search and analyze legal documents.

## Features

- **Simple Query Interface**: Clean, user-friendly design for entering legal questions
- **Real-time Processing**: Live status updates and loading states
- **Query History**: Automatic saving of recent queries with quick replay
- **Health Monitoring**: API status checking with visual indicators
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Error Handling**: Clear error messages and recovery options

## Architecture

```
ui/
├── backend/
│   └── api.py          # FastAPI server providing REST endpoints
├── frontend/
│   ├── index.html      # Main HTML structure
│   ├── style.css       # Modern, responsive styling
│   └── script.js       # Frontend logic and API integration
└── tests/
    └── (test files)    # Unit and integration tests
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js (optional, for serving frontend)
- All dependencies from main project requirements.txt

### Backend Setup

1. Install additional dependencies:
```bash
pip install fastapi uvicorn[standard] python-dotenv
```

2. Ensure environment variables are configured in `.env`:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
ENVIRONMENT=development

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
LANGCHAIN_PROJECT=legal-retriever-api
```

### Running the Application

1. Start the backend API:
```bash
cd ui/backend
python api.py
```

The API will be available at `http://localhost:8080`
API documentation at `http://localhost:8080/api/docs`

2. Serve the frontend (choose one method):

**Option A: Python HTTP Server**
```bash
cd ui/frontend
python -m http.server 8000
```

**Option B: Node.js Live Server**
```bash
npx live-server ui/frontend --port=8000
```

**Option C: Direct File Access**
Simply open `ui/frontend/index.html` in your browser

3. Access the UI at `http://localhost:8000`

## Usage

1. **Enter a Query**: Type your legal question in the text area
2. **Select Search Depth**: Choose iteration count (optional)
3. **Submit**: Click "Search Documents" or press Enter
4. **View Results**: The answer will appear below with metadata
5. **History**: Click any previous query to reload it

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and agent availability

### Query Processing
```
POST /api/query
Content-Type: application/json

{
    "query": "What are the damages claimed?",
    "max_iterations": 3,
    "request_id": "optional-tracking-id"
}
```

### Statistics
```
GET /api/stats
```
Returns usage statistics and performance metrics

## Development

### Frontend Development

The frontend uses vanilla JavaScript for simplicity. Key files:

- `index.html`: Semantic HTML5 structure with accessibility features
- `style.css`: CSS custom properties for easy theming
- `script.js`: Modern ES6+ JavaScript with async/await

### Backend Development

The backend uses FastAPI for high performance and automatic documentation:

- Pydantic models for request/response validation
- Async request handling with timeouts
- Comprehensive error handling and logging
- LangSmith integration for tracing

### Testing

Run tests from the project root:

```bash
# Backend tests
pytest ui/tests/test_api.py -v

# Integration tests
pytest ui/tests/test_integration.py -v
```

## Security Considerations

1. **CORS**: Configured for localhost only by default
2. **Input Validation**: Query length limits and sanitization
3. **Error Messages**: Sensitive information filtered
4. **Rate Limiting**: Can be added via middleware
5. **Authentication**: Can be added via API keys

## Performance Optimization

1. **Agent Initialization**: Single instance shared across requests
2. **Async Processing**: Non-blocking request handling
3. **Timeout Protection**: 2-minute timeout for long queries
4. **Response Caching**: Can be added for identical queries
5. **Frontend Optimization**: Debounced inputs, efficient DOM updates

## Troubleshooting

### API Won't Start
- Check if port 8080 is available
- Verify all project dependencies are installed
- Check logs for initialization errors

### CORS Errors
- Ensure frontend is served from allowed origin
- Check CORS configuration in api.py

### Slow Responses
- Check agent configuration and max_iterations
- Monitor API stats endpoint
- Consider implementing caching

## Future Enhancements

1. **Streaming Responses**: Show results as they're generated
2. **Source Citations**: Display specific documents referenced
3. **Query Suggestions**: Auto-complete based on history
4. **Advanced Filters**: Category and date range selection
5. **Export Options**: Save results as PDF or Markdown

## Contributing

When adding features:
1. Maintain simplicity - avoid unnecessary dependencies
2. Ensure responsive design
3. Add appropriate error handling
4. Include tests for new functionality
5. Update this README

## License

Same as parent project