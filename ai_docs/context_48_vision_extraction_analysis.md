# Context 48: Vision Text Extraction Analysis - Why It's Not Working

## Executive Summary

The vision-based text extraction capability was designed and documented but **never fully implemented**. The current system has placeholders for vision OCR but defaults to PyPDF text extraction, which fails on scanned documents. With the Google Gemma vision models now available, we need to activate this functionality.

## Investigation Findings

### 1. Original Design (Context 4)

The original implementation plan included comprehensive vision-based OCR:

```python
# From context_4_vector_search_implementation.md
class VisionOCRExtractor:
    def __init__(self, model="gemma3:12b", dpi=300):
        self.model = model
        self.dpi = dpi
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        # Convert PDF pages to images
        images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
        
        # Use vision model for OCR
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            images=[img_base64],
            options={"temperature": 0.1}
        )
```

**Key Features Designed:**
- PDF to image conversion at 300 DPI
- Vision model OCR using Gemma
- Structured text extraction with layout preservation
- Semantic chunking based on document structure

### 2. Current Implementation Status

#### In `vector_processor.py` (Lines 71-75):
```python
# If text is too short and vision OCR is enabled, try vision model
if self.enable_vision_ocr and len(text.strip()) < 50 and self.vision_model:
    logger.info(f"Page {page_num + 1} has minimal text, attempting vision OCR...")
    # Note: This is a placeholder - actual vision OCR implementation would go here
    # For now, we'll use the PyPDF text
```

**This is just a placeholder!** The actual vision OCR was never implemented.

#### In `config.py`:
```python
ENABLE_VISION_OCR = os.getenv("ENABLE_VISION_OCR", "false").lower() == "true"
VISION_OCR_MODEL = os.getenv("VISION_OCR_MODEL", "llava")
```

**Default is `false`**, so vision OCR is disabled by default.

### 3. Why Vision Extraction Failed

1. **Not Implemented**: The core vision extraction logic was never written - only placeholders exist
2. **Disabled by Default**: `ENABLE_VISION_OCR` defaults to `false`
3. **Wrong Model**: Default vision model is "llava", not the available Gemma models
4. **Missing Dependencies**: `pdf2image` is in requirements.txt but the conversion code isn't implemented

### 4. Available Gemma Vision Models

From the current configuration:
```
google/gemma-3-4b    (3.03 GB) - Vision capable
google/gemma-3-12b   (8.07 GB) - Vision capable  
google/gemma-3-27b   (16.87 GB) - Vision capable
```

All three Gemma models support vision tasks and can perform OCR.

## Root Cause Analysis

### Document 1 (SOUFI depo.pdf) - Only Bates Stamps Extracted

**What Happened:**
- PyPDF extracted only 682 characters from 57 pages (12 chars/page)
- Only Bates stamp text was found: "TEST000001\n", "TEST000002\n", etc.
- Document is likely a scanned PDF with no text layer

**Why Vision OCR Didn't Help:**
1. Vision OCR is disabled (`ENABLE_VISION_OCR=false`)
2. Even if enabled, the implementation is missing
3. The placeholder code would still use PyPDF text

### Document 2 (Alrashedi) - Successful Extraction

**What Happened:**
- PyPDF successfully extracted 68,063 characters
- This is a text-based PDF (likely a digital deposition transcript)
- No vision OCR needed

### Document 3 (Application) - Complete Failure

**What Happened:**
- PyPDF couldn't read the file at all (0 bytes extracted)
- Possible PDF corruption or unusual format
- Vision OCR could potentially help here

## Solution: Implement Vision Text Extraction

### 1. Update Configuration

```python
# In .env or config.py
ENABLE_VISION_OCR = "true"  # Enable by default
VISION_OCR_MODEL = "google/gemma-3-12b"  # Use available Gemma model
PDF_DPI = "300"  # High quality for OCR
```

### 2. Implement the Vision Extraction

```python
# In vector_processor.py, replace the placeholder:

def _extract_with_vision(self, page, page_num: int) -> str:
    """Extract text from page image using Gemma vision model."""
    try:
        # Convert page to image
        import pdf2image
        import io
        import base64
        
        # Convert single page to image
        images = pdf2image.convert_from_bytes(
            page.extract_image_data(),
            dpi=300,
            first_page=page_num,
            last_page=page_num
        )
        
        if not images:
            return ""
            
        # Convert to base64
        img_buffer = io.BytesIO()
        images[0].save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Use Gemma vision model for OCR
        if self.llm_provider == "lmstudio":
            response = self.lmstudio_client.generate(
                model="google/gemma-3-12b",
                prompt="Extract all text from this document image. Return only the text content, preserving the original formatting and layout.",
                images=[img_base64],
                temperature=0.1
            )
        else:
            # Ollama fallback
            response = self.ollama_client.generate(
                model=self.vision_model,
                prompt="Extract all text from this document image. Return only the text content, preserving the original formatting and layout.",
                images=[img_base64],
                options={"temperature": 0.1}
            )
        
        return response.get('response', '')
        
    except Exception as e:
        logger.error(f"Vision OCR failed for page {page_num}: {e}")
        return ""
```

### 3. Modify Text Extraction Logic

```python
# In PDFToLangChainLoader.load():

for page_num in range(total_pages):
    page = pdf_reader.pages[page_num]
    
    # Extract text using PyPDF first
    text = page.extract_text()
    
    # Check if we need vision OCR
    if self.enable_vision_ocr and len(text.strip()) < 100:
        logger.info(f"Page {page_num + 1} has minimal text ({len(text)} chars), using vision OCR...")
        vision_text = self._extract_with_vision(page, page_num + 1)
        
        if vision_text and len(vision_text.strip()) > len(text.strip()):
            text = vision_text
            logger.info(f"Vision OCR extracted {len(vision_text)} characters")
```

### 4. Document Type Detection

Integrate with `DocumentAnalyzer` to automatically use vision for scanned documents:

```python
# In document_orchestrator.py
if self.document_analyzer:
    analysis = self.document_analyzer.analyze_document(pdf_path)
    
    if analysis.get("is_scanned") or analysis.get("needs_ocr"):
        # Force vision OCR for this document
        loader = PDFToLangChainLoader(
            str(pdf_path),
            enable_vision_ocr=True,
            vision_model="google/gemma-3-12b"
        )
```

## Implementation Priority

### Immediate Actions:

1. **Enable Vision OCR by Default**
   ```bash
   # In .env
   ENABLE_VISION_OCR=true
   VISION_OCR_MODEL=google/gemma-3-12b
   ```

2. **Implement Vision Extraction** in `vector_processor.py`
   - Replace placeholder with actual implementation
   - Use google/gemma-3-12b for vision tasks

3. **Test on Problem Documents**
   - Re-process "3.14.25 SOUFI depo.pdf" with vision OCR
   - Should extract actual deposition content, not just Bates stamps

### Expected Results After Implementation:

- **Document 1**: Full deposition text extracted (not just Bates stamps)
- **Document 2**: No change (already working)
- **Document 3**: Potential recovery if PDF is readable as image

## Performance Considerations

With Gemma vision models:
- **google/gemma-3-4b**: Fast, suitable for simple documents (~1-2 sec/page)
- **google/gemma-3-12b**: Balanced, good for most legal documents (~3-5 sec/page)
- **google/gemma-3-27b**: Best quality for complex layouts (~5-10 sec/page)

Recommendation: Use `google/gemma-3-12b` as default for good balance.

## Testing the Fix

```python
# Quick test script
from src.vector_processor import PDFToLangChainLoader

# Enable vision OCR
loader = PDFToLangChainLoader(
    "test_sample_input/3.14.25 SOUFI depo.pdf",
    enable_vision_ocr=True,
    vision_model="google/gemma-3-12b"
)

docs = loader.load()
for i, doc in enumerate(docs[:3]):
    print(f"Page {i+1}: {len(doc.page_content)} chars")
    print(f"Preview: {doc.page_content[:200]}...")
```

## Conclusion

The vision text extraction capability exists in design but **not in implementation**. The system has been falling back to PyPDF, which cannot extract text from scanned documents. With the Gemma vision models now available, we need to:

1. Enable vision OCR by default
2. Implement the actual vision extraction code  
3. Use `google/gemma-3-12b` as the default vision model
4. Integrate with DocumentAnalyzer for automatic detection

This will resolve the issue where scanned PDFs (like Document 1) only show Bates stamps instead of actual content.