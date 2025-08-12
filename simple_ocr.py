# simple_ocr.py - Standalone OCR using paddleocr or easyocr
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class SimpleOCR:
    """Simple OCR engine using PaddleOCR or EasyOCR as fallback"""
    
    def __init__(self):
        self.ocr_engine = None
        self.engine_type = None
        
        # Try PaddleOCR first (better for multilingual)
        try:
            from paddleocr import PaddleOCR
            # Initialize with English (works well for Hebrew text too)
            self.ocr_engine = PaddleOCR(
                use_textline_orientation=True, 
                lang='en'  # English works well for Hebrew characters
            )
            self.engine_type = "paddleocr"
            print("✅ PaddleOCR initialized (Multilingual support)")
        except ImportError:
            print("⚠️ PaddleOCR not available, trying EasyOCR...")
            
            # Fallback to EasyOCR
            try:
                import easyocr
                self.ocr_engine = easyocr.Reader(['he', 'en'])  # Hebrew and English
                self.engine_type = "easyocr"
                print("✅ EasyOCR initialized (Hebrew + English support)")
            except ImportError:
                print("❌ No OCR engine available. Install with:")
                print("   pip install paddleocr  # Recommended")
                print("   or")
                print("   pip install easyocr")
                self.ocr_engine = None
    
    def is_available(self) -> bool:
        """Check if OCR engine is available"""
        return self.ocr_engine is not None
    
    def extract_text_from_image(self, image_path_or_array) -> List[Tuple[str, float]]:
        """
        Extract text from image
        Returns list of (text, confidence) tuples
        """
        if not self.is_available():
            return []
        
        try:
            # Handle both file paths and numpy arrays
            if isinstance(image_path_or_array, (str, Path)):
                image = cv2.imread(str(image_path_or_array))
            else:
                image = image_path_or_array
            
            if image is None:
                print("❌ Could not load image")
                return []
            
            results = []
            
            if self.engine_type == "paddleocr":
                # PaddleOCR returns [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                ocr_results = self.ocr_engine.ocr(image)
                
                if ocr_results and ocr_results[0]:
                    for line in ocr_results[0]:
                        if line:
                            text = line[1][0]
                            confidence = line[1][1]
                            results.append((text, confidence))
            
            elif self.engine_type == "easyocr":
                # EasyOCR returns [([x1,y1,x2,y2], text, confidence)]
                ocr_results = self.ocr_engine.readtext(image)
                
                for result in ocr_results:
                    text = result[1]
                    confidence = result[2]
                    results.append((text, confidence))
            
            return results
            
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return []
    
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int) -> str:
        """Extract text from a specific PDF page using OCR"""
        try:
            import fitz  # PyMuPDF
            
            pdf_doc = fitz.open(pdf_path)
            if page_num >= len(pdf_doc):
                pdf_doc.close()
                return ""
            
            page = pdf_doc[page_num]
            # Convert to image with high resolution for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.pil_tobytes(format="PNG")
            
            # Convert to OpenCV format
            import io
            from PIL import Image
            pil_image = Image.open(io.BytesIO(img_data))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            pdf_doc.close()
            
            # Extract text using OCR
            results = self.extract_text_from_image(cv_image)
            
            # Combine text with confidence threshold
            text_parts = [text for text, conf in results if conf > 0.3]
            return "\\n".join(text_parts)
            
        except Exception as e:
            print(f"❌ PDF page OCR failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from entire PDF using OCR"""
        try:
            import fitz  # PyMuPDF
            
            pdf_doc = fitz.open(pdf_path)
            all_text_parts = []
            
            print(f"   🔍 Processing {len(pdf_doc)} pages with {self.engine_type.upper()}...")
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                # Convert to image with high resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.pil_tobytes(format="PNG")
                
                # Convert to OpenCV format
                import io
                from PIL import Image
                pil_image = Image.open(io.BytesIO(img_data))
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Extract text using OCR
                results = self.extract_text_from_image(cv_image)
                
                # Combine text with confidence threshold
                page_text_parts = [text for text, conf in results if conf > 0.3]
                
                if page_text_parts:
                    page_text = "\\n".join(page_text_parts)
                    all_text_parts.append(f"\\n--- Page {page_num + 1} ---\\n{page_text}")
                    print(f"   ✅ OCR extracted {len(page_text)} chars from page {page_num + 1}")
                else:
                    print(f"   ⚠️ No text detected on page {page_num + 1}")
            
            pdf_doc.close()
            
            if all_text_parts:
                full_text = "\\n".join(all_text_parts)
                print(f"   🎯 OCR processing complete: {len(full_text)} total characters from {len(all_text_parts)} pages")
                return full_text
            else:
                return "No text detected in PDF pages via OCR"
                
        except ImportError:
            print("   ⚠️ PyMuPDF (fitz) not available for PDF-to-image conversion")
            return "PDF OCR failed: PyMuPDF not installed"
        except Exception as e:
            print(f"   ❌ PDF OCR processing failed: {e}")
            return f"PDF OCR failed: {str(e)}"

# Test function
def test_ocr():
    """Test the OCR functionality"""
    ocr = SimpleOCR()
    if ocr.is_available():
        print(f"OCR engine ready: {ocr.engine_type}")
        return True
    else:
        print("No OCR engine available")
        return False

if __name__ == "__main__":
    test_ocr()
