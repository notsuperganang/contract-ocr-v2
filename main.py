#!/usr/bin/env python3
"""
Telkom Contract Data Extraction System
Main CLI interface for the PP-StructureV3-based contract extraction pipeline

Usage:
    python main.py --input contract.pdf --output results.xlsx
    python main.py --input contracts/ --output outputs/ --batch
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Telkom Contract Data Extraction using PP-StructureV3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single contract
  python main.py --input contract.pdf --output results.xlsx
  
  # Process multiple contracts in batch
  python main.py --input contracts/ --output outputs/ --batch
  
  # Use custom configuration
  python main.py --input contract.pdf --output results.xlsx --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input PDF contract file or directory containing contracts"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        required=True,
        help="Output Excel file path or directory for batch processing"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration file (default: config/pipeline_config.yaml)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch processing mode for directory input"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual processing"
    )
    
    args = parser.parse_args()
    
    # Validate input and output paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)
    
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)
        
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' does not exist")
        sys.exit(1)
    
    # Import the extractor
    try:
        from src.pipeline.extractor import TelkomContractExtractor
        from src.utils.logger import initialize_logging, get_logger
        from src.utils.config import load_config
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    print("Telkom Contract Data Extraction System")
    print("=====================================")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path}")
    print(f"Batch mode: {args.batch}")
    print(f"Verbose: {args.verbose}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would process contracts with the above configuration")
        return
    
    # Initialize logging
    log_level = "DEBUG" if args.verbose else "INFO"
    config_manager = load_config(str(config_path))
    config_manager.logging.level = log_level
    initialize_logging(config_manager)
    
    logger = get_logger("main")
    logger.info("Starting Telkom Contract Data Extraction")
    
    try:
        # Initialize extractor
        logger.info("Initializing extractor...")
        extractor = TelkomContractExtractor(str(config_path))
        
        if args.batch and input_path.is_dir():
            # Batch processing
            logger.info(f"Starting batch processing: {input_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = extractor.batch_extract(input_path, output_path)
            
            # Print summary
            successful = sum(1 for r in results if r.success)
            print(f"\n=== Batch Processing Summary ===")
            print(f"Total files: {len(results)}")
            print(f"Successful extractions: {successful}")
            print(f"Success rate: {successful/len(results)*100:.1f}%")
            print(f"Results saved to: {output_path}")
            
        else:
            # Single file processing
            logger.info(f"Processing single file: {input_path}")
            
            result = extractor.extract_from_file(input_path)
            
            if result.success:
                # Save results
                if output_path.suffix.lower() == '.json':
                    result.contract_data.save_to_json(str(output_path))
                    logger.info(f"Results saved to: {output_path}")
                else:
                    # Default to JSON with .json extension
                    json_output = output_path.with_suffix('.json')
                    result.contract_data.save_to_json(str(json_output))
                    logger.info(f"Results saved to: {json_output}")
                
                # Print summary
                summary = result.contract_data.get_extraction_summary()
                print(f"\n=== Extraction Summary ===")
                print(f"Document: {summary['document_name']}")
                print(f"Fields extracted: {summary['successful_extractions']}/{summary['total_fields']}")
                print(f"Success rate: {summary['success_rate']:.1f}%")
                print(f"Processing time: {summary['processing_time']:.2f}s")
                print(f"Overall confidence: {summary['overall_confidence']:.2f}")
                
                if result.warnings:
                    print(f"\nWarnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
                
            else:
                print(f"\n❌ Extraction failed: {result.error_message}")
                sys.exit(1)
        
        # Cleanup
        extractor.cleanup()
        logger.info("Extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()