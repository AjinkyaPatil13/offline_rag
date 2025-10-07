#!/usr/bin/env python3
"""
Test OCR functionality to verify text extraction from code images.
"""

import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# Add the rag module to path
sys.path.append(str(Path(__file__).parent))

from rag.ingestion import _extract_text_from_image

def create_code_image(path: Path) -> Path:
    """Create a simple code image for testing"""
    # Create a white background
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("consola.ttf", 16)  # Windows Consolas
    except:
        try:
            font = ImageFont.truetype("Monaco.ttf", 16)  # Mac Monaco
        except:
            font = ImageFont.load_default()  # Default fallback
    
    # Sample code text
    code_lines = [
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1"
    ]
    
    # Draw the code
    y_position = 30
    for line in code_lines:
        draw.text((30, y_position), line, fill='black', font=font)
        y_position += 25
    
    # Save the image
    img.save(path)
    return path

def test_ocr():
    """Test OCR text extraction"""
    print("🧪 Testing OCR functionality...\n")
    
    # Create a test code image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        code_img_path = create_code_image(temp_path / "test_code.png")
        
        print(f"📸 Created test code image: {code_img_path.name}")
        print(f"🔍 Extracting text using OCR...\n")
        
        # Extract text
        extracted_text = _extract_text_from_image(code_img_path)
        
        if extracted_text.strip():
            print("✅ OCR extraction successful!")
            print(f"📝 Extracted text:\n")
            print("=" * 50)
            print(extracted_text)
            print("=" * 50)
            
            # Check if it looks like code
            code_indicators = ['def ', 'if ', 'return', '==', '<=', '>=', '//', '(', ')']
            found_indicators = [ind for ind in code_indicators if ind in extracted_text]
            
            if found_indicators:
                print(f"🎉 Code detected! Found indicators: {found_indicators}")
                return True
            else:
                print("⚠️ Text extracted, but doesn't look like code")
                return False
        else:
            print("❌ No text extracted by OCR")
            return False

def test_with_your_image():
    """Test with your actual binary search image if available"""
    # Look for common image files in current directory
    current_dir = Path(".")
    image_files = []
    
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(current_dir.glob(ext))
    
    if image_files:
        print(f"\n🔍 Found {len(image_files)} images in current directory:")
        for img in image_files[:5]:  # Show first 5
            print(f"  - {img.name}")
        
        # Test with first image
        test_img = image_files[0]
        print(f"\n🧪 Testing OCR on: {test_img.name}")
        
        extracted_text = _extract_text_from_image(test_img)
        
        if extracted_text.strip():
            print("✅ OCR extraction successful!")
            print(f"📝 Extracted text (first 500 chars):\n")
            print("=" * 50)
            print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
            print("=" * 50)
            return True
        else:
            print("❌ No text extracted from your image")
            return False
    else:
        print("\n📁 No images found in current directory")
        return False

def main():
    print("🚀 OCR Test Suite\n")
    print("This will test EasyOCR text extraction from code images.\n")
    
    try:
        # Test 1: Synthetic code image
        print("=" * 60)
        print("TEST 1: Synthetic Code Image")
        print("=" * 60)
        success1 = test_ocr()
        
        # Test 2: Real images in directory
        print("\n" + "=" * 60)
        print("TEST 2: Real Images in Directory")
        print("=" * 60)
        success2 = test_with_your_image()
        
        print(f"\n📊 Results:")
        print(f"  Synthetic image: {'✅ PASS' if success1 else '❌ FAIL'}")
        print(f"  Real image: {'✅ PASS' if success2 else '❌ N/A'}")
        
        if success1:
            print(f"\n🎉 OCR is working! Your app should now be able to:")
            print(f"  - Extract text from code images")
            print(f"  - Include code content in search results")  
            print(f"  - Answer questions about code in images")
        else:
            print(f"\n❌ OCR needs troubleshooting")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        print("❌ There may be issues with the OCR setup")

if __name__ == "__main__":
    main()