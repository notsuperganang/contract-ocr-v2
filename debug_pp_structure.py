#!/usr/bin/env python3
"""
Debug script untuk menganalisis hasil PP-StructureV3
Membantu memahami struktur data yang dikembalikan
"""

import json
import sys
from pathlib import Path
from paddleocr import PPStructureV3
from src.pipeline.preprocessor import DocumentPreprocessor
from src.utils.config import ConfigManager

def debug_pp_structure_results(pdf_path: str):
    """Debug PP-StructureV3 results structure"""
    
    print("🔍 DEBUG: PP-StructureV3 Results Analysis")
    print("=" * 50)
    
    # Initialize components
    config = ConfigManager("config/pipeline_config.yaml")
    model_config = config.get_model_config_dict()
    
    preprocessor = DocumentPreprocessor()
    pp_structure = PPStructureV3(**model_config)
    
    # Process document
    print(f"📄 Processing: {pdf_path}")
    images = preprocessor.process_document(pdf_path)
    print(f"✅ Extracted {len(images)} pages")
    
    # Process first page only for debugging
    if images:
        print(f"\n🔬 Analyzing page 1 structure...")
        image = images[0]
        
        # Run PP-StructureV3
        page_results = pp_structure.predict(image)
        print(f"📊 Got {len(page_results)} results from PP-StructureV3")
        
        # Analyze each result
        for i, result in enumerate(page_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Type: {type(result)}")
            
            # Check available attributes
            if hasattr(result, '__dict__'):
                attrs = list(result.__dict__.keys())
                print(f"Attributes: {attrs}")
            else:
                # Try dir() for other types
                attrs = [attr for attr in dir(result) if not attr.startswith('_')]
                print(f"Public methods/attrs: {attrs[:10]}...")  # Show first 10
            
            # Check for json property
            if hasattr(result, 'json'):
                print(f"📋 Has 'json' property")
                try:
                    json_data = result.json
                    print(f"JSON keys: {list(json_data.keys()) if isinstance(json_data, dict) else 'Not a dict'}")
                    
                    # Save detailed JSON for inspection
                    debug_file = f"debug_result_{i+1}.json"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
                    print(f"💾 Saved detailed JSON to: {debug_file}")
                    
                except Exception as e:
                    print(f"❌ Error accessing json: {e}")
            
            # Check for overall_ocr_res
            if hasattr(result, 'overall_ocr_res'):
                ocr_res = result.overall_ocr_res
                print(f"📝 Has 'overall_ocr_res'")
                if ocr_res:
                    print(f"OCR keys: {list(ocr_res.keys()) if isinstance(ocr_res, dict) else 'Not a dict'}")
                    
                    if isinstance(ocr_res, dict) and 'rec_texts' in ocr_res:
                        texts = ocr_res['rec_texts']
                        print(f"📚 Found {len(texts)} text items")
                        
                        # Show first few texts
                        for j, text in enumerate(texts[:5]):
                            print(f"  Text {j+1}: '{text[:50]}...' (length: {len(text)})")
                else:
                    print("⚠️  overall_ocr_res is None or empty")
            
            # Check for table_res_list
            if hasattr(result, 'table_res_list'):
                table_res = result.table_res_list
                print(f"🔳 Has 'table_res_list'")
                if table_res:
                    print(f"📊 Found {len(table_res)} table results")
                    for j, table in enumerate(table_res):
                        if isinstance(table, dict):
                            table_keys = list(table.keys())
                            print(f"  Table {j+1} keys: {table_keys}")
                            
                            if 'pred_html' in table:
                                html_content = table['pred_html']
                                print(f"  HTML length: {len(html_content)} chars")
                else:
                    print("⚠️  table_res_list is None or empty")
            
            # Check for layout detection results
            if hasattr(result, 'layout_det_res'):
                layout_res = result.layout_det_res
                print(f"🏗️  Has 'layout_det_res'")
                if layout_res:
                    print(f"Layout keys: {list(layout_res.keys()) if isinstance(layout_res, dict) else 'Not a dict'}")
                else:
                    print("⚠️  layout_det_res is None or empty")
            
            print("-" * 30)
    
    # Cleanup
    preprocessor.cleanup()
    
    print("\n✅ Debug analysis completed!")
    print("Check the debug_result_*.json files for detailed structure")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_pp_structure.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    debug_pp_structure_results(pdf_path)