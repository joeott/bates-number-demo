# Context 44: Multi-Model Pipeline Test Flow for Recamier v. YMCA

## Executive Summary

This document provides a comprehensive test plan for the multi-model pipeline using the Recamier v. YMCA case documents. The test leverages Google Gemma vision models (3-4b, 3-12b, 3-27b) and Snowflake Arctic embeddings to process diverse legal documents through each pipeline stage.

## Model Assignment Strategy

Based on the available Gemma models (all vision-capable):

1. **Visual Analysis**: `google/gemma-3-4b` (3.03 GB)
   - Fastest model for initial document layout analysis
   - Sufficient for structure detection and quality assessment

2. **Reasoning/Entity Extraction**: `google/gemma-3-12b` (8.07 GB)
   - Mid-size model balancing speed and reasoning capability
   - Better for complex entity relationships and temporal reasoning

3. **Categorization**: `google/gemma-3-4b` (3.03 GB)
   - Fast categorization doesn't require large model
   - Quick processing for high-volume documents

4. **Synthesis/Summary**: `google/gemma-3-27b` (16.87 GB)
   - Largest model for complex synthesis and summarization
   - Best thinking capability for nuanced legal analysis

5. **Embeddings**: `text-embedding-snowflake-arctic-embed-l-v2.0`
   - Dedicated embedding model for semantic search

## Test Document Selection

### Document Diversity Analysis

The Recamier v. YMCA folder contains:
- **Medical Records**: 1000+ pages (various providers)
- **Depositions**: Multiple video and text transcripts
- **Legal Filings**: Motions, responses, court orders
- **Discovery Documents**: Interrogatories, production requests
- **Financial Documents**: Bills, invoices, payment records
- **Correspondence**: Letters, emails, notices
- **Media Files**: Videos (.mov, .mp4), images
- **Administrative**: Service documents, certificates

### Selected Test Documents

#### Stage 1: Visual Analysis Test Set
1. **Complex Layout**: `Filings/Motions in Limine/Exhibit 1.pdf`
   - Tests: Multi-column detection, exhibit stickers, stamps
2. **Poor Quality Scan**: `Client Docs/Nikita Notes/Nikita Letter.pdf`
   - Tests: Handwriting detection, OCR quality assessment
3. **Medical Form**: `Medicals/Encounter Summary for Nikita Recamier.pdf`
   - Tests: Table extraction, form field detection
4. **Multi-page Filing**: `Filings/Nikita Recamier - Motions in Limine.pdf`
   - Tests: Page relationships, continuous document detection

#### Stage 2: Reasoning Test Set
1. **Deposition**: `Depositions/Nikita Recamier Depo/11748557 Recamier.Nikita 092524.full.pdf`
   - Tests: Speaker identification, timeline extraction, cross-references
2. **Discovery Response**: `Discovery/D Responses/RECAMIER N GRYMCAs Answers to Plaintiffs First Interrogatories (1).pdf`
   - Tests: Claim-response mapping, entity relationships
3. **Medical Records**: `Client Docs/RecaimerWors v. YMCA- Records via authorization 10.27.23/Wors Med Rec Dr. Laurence Kinsella (LK 1- 304) (via spvg auth).pdf`
   - Tests: Medical timeline, treatment relationships, provider network

#### Stage 3: Categorization Test Set
1. **Ambiguous Names**: 
   - `Dierker Check $1618.75. envelope.pdf` → Should categorize as "Bill"
   - `FAX_20241226_1735238190_746.pdf` → Requires content analysis
2. **Complex Titles**:
   - `RECAMIER_N DEFENDANTS' SECOND SUPPLEMENTAL OBJECTIONS AND RESPONSES TO PLAINTIFF.pdf`
   - `Nikita Recamier - Ds Memorandum in Support of Motion for Summary Judgement.pdf`

#### Stage 4: Synthesis Test Set
1. **Document Bundle**: All discovery responses in `Discovery/D Responses/`
   - Tests: Multi-document synthesis, contradiction detection
2. **Expert Depositions**: All doctor depositions
   - Tests: Medical opinion synthesis, expert testimony comparison

## Detailed Test Flow

### Pre-Test Setup

```python
# Configuration update for Gemma models
LMSTUDIO_VISUAL_MODEL=google/gemma-3-4b
LMSTUDIO_REASONING_MODEL=google/gemma-3-12b
LMSTUDIO_CATEGORIZATION_MODEL=google/gemma-3-4b
LMSTUDIO_SYNTHESIS_MODEL=google/gemma-3-27b
LMSTUDIO_EMBEDDING_MODEL=text-embedding-snowflake-arctic-embed-l-v2.0
ENABLE_MULTI_MODEL=true
```

### Stage 1: Visual Analysis (gemma-3-4b)

**Test Procedure**:
1. Load test document
2. Execute visual analysis
3. Verify outputs

**Verification Criteria**:
- ✓ Document type identification (scanned vs. native PDF)
- ✓ Layout structure detection (headers, footers, columns)
- ✓ Visual element identification (signatures, stamps, tables)
- ✓ Quality score (0.0-1.0 scale)
- ✓ Page relationship detection

**Expected Outputs**:
```json
{
  "document": "Motions in Limine/Exhibit 1.pdf",
  "layout_type": "formal_legal_exhibit",
  "visual_elements": ["exhibit_sticker", "page_numbers", "court_header"],
  "quality_score": 0.85,
  "structural_segments": ["header", "body", "footer"],
  "page_relationships": "exhibit_attachment",
  "ocr_needed": false
}
```

**Error Scenarios**:
- Corrupted PDF: Should return error gracefully
- Non-PDF file: Should skip with warning
- Empty pages: Should note and continue

### Stage 2: Entity Extraction & Reasoning (gemma-3-12b)

**Test Procedure**:
1. Provide document text + visual context
2. Execute reasoning analysis
3. Validate entity graph

**Verification Criteria**:
- ✓ All parties correctly identified
- ✓ Coreference resolution (pronouns → entities)
- ✓ Temporal events in correct sequence
- ✓ Cross-references validated
- ✓ Logical relationships mapped

**Expected Outputs**:
```json
{
  "entities": {
    "parties": {
      "plaintiff": "Nikita Recamier (aka Nikita Wors)",
      "defendant": "Gateway Region YMCA",
      "co_defendants": ["Individual Instructors"]
    },
    "attorneys": {
      "plaintiff_counsel": "Ott Law Firm",
      "defense_counsel": "SPG Law Firm"
    },
    "medical_providers": [
      "Dr. Laurence Kinsella",
      "Dr. Michael Stotler",
      "SSM Health"
    ],
    "key_dates": {
      "incident": "Date from YMCA class",
      "filing": "From petition",
      "depositions": ["09/25/24", "12/19/24", "01/21/25"]
    }
  },
  "relationships": [
    {"type": "treats", "subject": "Dr. Kinsella", "object": "Nikita Recamier"},
    {"type": "represents", "subject": "Ott Law Firm", "object": "Nikita Recamier"}
  ],
  "timeline": [
    {"event": "YMCA incident", "date": "extracted_date"},
    {"event": "Medical treatment begins", "date": "extracted_date"},
    {"event": "Lawsuit filed", "date": "extracted_date"}
  ]
}
```

**Error Scenarios**:
- Conflicting dates: Flag for human review
- Missing entities: Note gaps in extraction
- Circular references: Detect and break loops

### Stage 3: Categorization (gemma-3-4b)

**Test Procedure**:
1. Process filename + visual/reasoning context
2. Execute categorization
3. Verify accuracy

**Verification Criteria**:
- ✓ Correct primary category
- ✓ Appropriate subcategory
- ✓ Confidence score > 0.8
- ✓ Consistent with content analysis

**Test Cases**:
```
Input: "Nikita Recamier - Motions in Limine.pdf"
Expected: Category="Pleading", Subcategory="Motion"

Input: "Wors Med Rec Dr. Laurence Kinsella (LK 1- 304).pdf"
Expected: Category="Medical Record", Subcategory="Physician Records"

Input: "Dierker Check $1618.75. envelope.pdf"
Expected: Category="Bill", Subcategory="Payment"

Input: "video1797266593.mp4"
Expected: Category="Video", Subcategory="Deposition Video"
```

**Error Scenarios**:
- Ambiguous filenames: Use content analysis
- Missing extensions: Infer from content
- Corrupted metadata: Fallback to visual analysis

### Stage 4: Synthesis & Summary (gemma-3-27b)

**Test Procedure**:
1. Aggregate all prior analysis
2. Execute comprehensive synthesis
3. Generate strategic insights

**Verification Criteria**:
- ✓ Accurate executive summary
- ✓ Key legal issues identified
- ✓ Document relationships mapped
- ✓ Contradictions noted
- ✓ Action items extracted

**Expected Outputs**:
```json
{
  "executive_summary": "Personal injury case arising from martial arts class at YMCA. Plaintiff alleges negligent instruction leading to injuries requiring extensive medical treatment.",
  
  "key_legal_issues": [
    "Negligent instruction/supervision",
    "Premises liability",
    "Assumption of risk defense",
    "Medical damages calculation"
  ],
  
  "document_relationships": {
    "core_pleadings": ["Petition", "Answer", "Motions"],
    "supporting_medical": ["Dr. Kinsella records", "SSM Health records"],
    "discovery_pairs": {
      "interrogatories": ["P's First", "D's Responses"],
      "document_requests": ["P's RFPs", "D's Productions"]
    }
  },
  
  "strategic_insights": {
    "strengths": [
      "Extensive medical documentation",
      "Multiple treating physicians"
    ],
    "weaknesses": [
      "Possible waiver/release defense",
      "Assumption of risk in martial arts"
    ],
    "critical_depositions": [
      "Plaintiff testimony on incident",
      "Instructor on safety protocols"
    ]
  }
}
```

**Error Scenarios**:
- Incomplete document set: Note missing elements
- Conflicting narratives: Highlight discrepancies
- Complex medical terminology: Maintain accuracy

### Stage 5: Embedding Generation (snowflake-arctic)

**Test Procedure**:
1. Create intelligent chunks using prior analysis
2. Generate embeddings with metadata
3. Verify search quality

**Verification Criteria**:
- ✓ Semantic coherence of chunks
- ✓ Metadata preservation
- ✓ Appropriate chunk sizes
- ✓ Cross-document linking

**Test Queries**:
1. "Find all mentions of safety protocols at YMCA"
2. "What injuries did plaintiff sustain?"
3. "Timeline of medical treatment"
4. "Expert opinions on causation"

## Error Handling Matrix

| Error Type | Detection Method | Recovery Strategy | Logging |
|------------|-----------------|-------------------|---------|
| Model Timeout | 30s threshold | Retry 3x, then skip | ERROR log with document |
| Memory Overflow | Monitor usage | Reduce batch size | WARN log with metrics |
| Corrupt PDF | Exception catch | Mark as failed, continue | ERROR log with path |
| Missing Model | Pre-flight check | Fall back to default | WARN log with fallback |
| API Rate Limit | 429 response | Exponential backoff | INFO log with wait time |

## Performance Benchmarks

### Expected Processing Times
- **Visual Analysis**: 2-5 seconds per page
- **Reasoning**: 10-30 seconds per document
- **Categorization**: 1-3 seconds per document
- **Synthesis**: 30-60 seconds per document set
- **Embeddings**: 5-10 seconds per 100 chunks

### Memory Usage Targets
- **Peak Usage**: < 80GB (leaving 48GB buffer)
- **Model Loading**: Staggered to prevent spikes
- **Batch Sizes**: Adjusted based on document size

## Success Metrics

1. **Accuracy Metrics**
   - Categorization: > 95% accuracy
   - Entity Extraction: > 90% recall
   - Relationship Mapping: > 85% precision

2. **Performance Metrics**
   - Total Processing: < 5 minutes for 100 documents
   - Error Rate: < 2%
   - Recovery Success: > 98%

3. **Quality Metrics**
   - Summary Relevance: Human evaluation score > 4/5
   - Search Precision: > 80% for test queries
   - Cross-reference Accuracy: > 90%

## Test Execution Script

```python
# test_pipeline_recamier.py

import os
from pathlib import Path
from src.document_orchestrator import DocumentOrchestrator
from src.model_discovery import ModelDiscoveryService

# Set up test environment
os.environ['LLM_PROVIDER'] = 'lmstudio'
os.environ['ENABLE_MULTI_MODEL'] = 'true'
os.environ['LMSTUDIO_VISUAL_MODEL'] = 'google/gemma-3-4b'
os.environ['LMSTUDIO_REASONING_MODEL'] = 'google/gemma-3-12b'
os.environ['LMSTUDIO_CATEGORIZATION_MODEL'] = 'google/gemma-3-4b'
os.environ['LMSTUDIO_SYNTHESIS_MODEL'] = 'google/gemma-3-27b'

# Test specific documents
test_docs = [
    'Filings/Motions in Limine/Exhibit 1.pdf',
    'Depositions/Nikita Recamier Depo/11748557 Recamier.Nikita 092524.full.pdf',
    'Discovery/D Responses/RECAMIER N GRYMCAs Answers to Plaintiffs First Interrogatories (1).pdf'
]

# Run pipeline with monitoring
orchestrator = DocumentOrchestrator()
for doc in test_docs:
    print(f"\n{'='*60}")
    print(f"Processing: {doc}")
    print('='*60)
    
    try:
        results = orchestrator.process_document(
            Path(f"input_documents/Recamier v. YMCA/{doc}")
        )
        
        # Verify each stage
        verify_results(results)
        
    except Exception as e:
        print(f"ERROR: {e}")
        log_error(doc, e)
```

## Post-Test Analysis

1. **Review Logs**: Check for warnings and errors
2. **Validate Outputs**: Spot-check categorizations and summaries
3. **Performance Review**: Compare against benchmarks
4. **Quality Assessment**: Human review of synthesis outputs
5. **Search Testing**: Verify retrieval accuracy

## Conclusion

This comprehensive test flow ensures the multi-model pipeline correctly processes the diverse Recamier v. YMCA documents. The combination of Gemma vision models for different tasks optimizes the balance between performance and quality, with the smaller models handling routine tasks and the larger 27B model providing deep analysis where needed.