"""
Quantize Model Script

This script downloads and quantizes the Meta-Llama-3-8B model.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to download and quantize the model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download and quantize the Meta-Llama-3-8B model")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to model configuration file")
    parser.add_argument("--method", type=str, choices=["bitsandbytes", "llama.cpp", "gptq"], default=None, help="Quantization method")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=None, help="Quantization bits")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for the quantized model")
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get model configuration
        model_id = config["llama3"]["model_id"]
        model_path = args.output_dir or config["llama3"]["model_path"]
        
        # Get quantization configuration
        quant_method = args.method or config["llama3"]["quantization"]["method"]
        quant_bits = args.bits or config["llama3"]["quantization"]["bits"]
        
        # Create output directory
        os.makedirs(model_path, exist_ok=True)
        
        # Quantize model based on method
        if quant_method == "bitsandbytes":
            quantize_with_bitsandbytes(model_id, model_path, quant_bits)
        elif quant_method == "llama.cpp":
            quantize_with_llamacpp(model_id, model_path, quant_bits)
        elif quant_method == "gptq":
            group_size = config["llama3"]["quantization"].get("group_size", 128)
            quantize_with_gptq(model_id, model_path, quant_bits, group_size)
        else:
            logger.error(f"Unsupported quantization method: {quant_method}")
            sys.exit(1)
        
        logger.info(f"Model quantized and saved to {model_path}")
    
    except Exception as e:
        logger.error(f"Error quantizing model: {str(e)}")
        sys.exit(1)

def quantize_with_bitsandbytes(model_id: str, output_dir: str, bits: int):
    """
    Quantize the model using bitsandbytes.
    
    Args:
        model_id: The model ID.
        output_dir: Output directory for the quantized model.
        bits: Quantization bits (4 or 8).
    """
    logger.info(f"Quantizing {model_id} to {bits} bits using bitsandbytes")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load and quantize model
        if bits == 4:
            logger.info("Using 4-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            logger.info("Using 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model and tokenizer saved to {output_dir}")
    
    except ImportError:
        logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error quantizing with bitsandbytes: {str(e)}")
        raise

def quantize_with_llamacpp(model_id: str, output_dir: str, bits: int):
    """
    Quantize the model using llama.cpp.
    
    Args:
        model_id: The model ID.
        output_dir: Output directory for the quantized model.
        bits: Quantization bits (4 or 8).
    """
    logger.info(f"Quantizing {model_id} to {bits} bits using llama.cpp")
    
    try:
        import subprocess
        import tempfile
        
        # Create temporary directory for the original model
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the model in Hugging Face format
            logger.info(f"Downloading {model_id} to {temp_dir}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Clone llama.cpp repository
            llama_cpp_dir = os.path.join(output_dir, "llama.cpp")
            if not os.path.exists(llama_cpp_dir):
                logger.info("Cloning llama.cpp repository")
                subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
                    check=True
                )
            
            # Build llama.cpp
            logger.info("Building llama.cpp")
            subprocess.run(
                ["make", "-j"],
                cwd=llama_cpp_dir,
                check=True
            )
            
            # Convert the model to GGUF format
            logger.info("Converting model to GGUF format")
            subprocess.run(
                [
                    "python", "convert.py",
                    temp_dir
                ],
                cwd=llama_cpp_dir,
                check=True
            )
            
            # Quantize the model
            logger.info(f"Quantizing model to {bits} bits")
            quant_type = f"q{bits}_0"
            gguf_model = os.path.join(llama_cpp_dir, "models", "model.gguf")
            output_model = os.path.join(output_dir, f"llama3-8b-{quant_type}.gguf")
            
            subprocess.run(
                [
                    "python", "quantize.py",
                    gguf_model,
                    output_model,
                    quant_type
                ],
                cwd=llama_cpp_dir,
                check=True
            )
            
            logger.info(f"Quantized model saved to {output_model}")
            
            # Copy tokenizer
            tokenizer.save_pretrained(output_dir)
    
    except ImportError:
        logger.error("Required packages not installed")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in subprocess: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error quantizing with llama.cpp: {str(e)}")
        raise

def quantize_with_gptq(model_id: str, output_dir: str, bits: int, group_size: int):
    """
    Quantize the model using GPTQ.
    
    Args:
        model_id: The model ID.
        output_dir: Output directory for the quantized model.
        bits: Quantization bits (4 or 8).
        group_size: Group size for quantization.
    """
    logger.info(f"Quantizing {model_id} to {bits} bits using GPTQ with group size {group_size}")
    
    try:
        # Check if auto-gptq is installed
        try:
            import auto_gptq
        except ImportError:
            logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
            sys.exit(1)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load and quantize model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config={
                "bits": bits,
                "group_size": group_size,
                "desc_act": True
            },
            device_map="auto",
            trust_remote_code=True
        )
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model and tokenizer saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Error quantizing with GPTQ: {str(e)}")
        raise

if __name__ == "__main__":
    main()
