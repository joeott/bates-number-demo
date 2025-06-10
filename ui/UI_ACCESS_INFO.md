# Legal Document Retriever UI - Access Information

## 🟢 Status: Running Successfully

The UI is now running and connected to the vector store containing 13 documents from the Recamier v. YMCA case.

## 📍 Access Points

### Web Interface
- **Main UI**: http://localhost:8000
- Open this in your web browser to use the retrieval interface

### API Endpoints
- **Health Check**: http://localhost:8080/health
- **API Documentation**: http://localhost:8080/api/docs
- **Statistics**: http://localhost:8080/api/stats

## 🎯 Quick Test

Try these sample queries in the UI:
1. "What is this case about?"
2. "Who are the parties involved?"
3. "What are the key allegations?"
4. "What damages are being claimed?"
5. "What evidence supports the breach of contract claim?"

## 📊 Current Configuration

- **Vector Store**: 13 documents loaded from Recamier v. YMCA case
- **Search Method**: Hybrid (vector + PostgreSQL if available)
- **Rate Limiting**: 10 requests/minute, 100 requests/hour
- **Reranking**: Enabled (using BGE model if available)
- **Compression**: Temporarily disabled

## 🛑 Stopping the Servers

To stop both servers, run:
```bash
pkill -f 'python.*api.py|http.server'
```

## 🔧 Troubleshooting

If you encounter issues:

1. **Check API Health**: 
   ```bash
   curl http://localhost:8080/health
   ```

2. **View API Logs**:
   ```bash
   tail -f api.log
   ```

3. **Restart Servers**:
   ```bash
   # Stop existing servers
   pkill -f 'python.*api.py|http.server'
   
   # Start API
   cd ui/backend && python api.py
   
   # In another terminal, start frontend
   cd ui/frontend && python -m http.server 8000
   ```

## ✅ Verified Functionality

- API health endpoint responding
- Agent successfully initialized
- Vector store connected and searchable
- Query processing working (10s average response time)
- Frontend accessible at port 8000
- CORS properly configured

The system is ready for use!