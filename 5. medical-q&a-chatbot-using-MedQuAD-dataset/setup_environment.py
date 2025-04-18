import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=None):
    """Run a shell command and log the output"""
    if description:
        logger.info(f"{description}...")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            for line in process.stderr:
                print(line.strip(), file=sys.stderr)
            logger.error(f"Command failed with return code {process.returncode}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def setup_environment(args):
    """Set up the environment with the correct dependencies"""
    # Step 1: Uninstall problematic packages
    if args.clean:
        logger.info("Cleaning existing packages...")
        packages_to_remove = [
            "numpy",
            "scipy",
            "pandas",
            "scikit-learn",
            "sentence-transformers",
            "faiss-cpu",
            "spacy",
            "scispacy",
            "flask",
            "flask-cors",
            "werkzeug"
        ]
        
        for package in packages_to_remove:
            run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")
    
    # Step 2: Install core dependencies in the correct order
    logger.info("Installing core dependencies...")
    
    # Install NumPy first (specific version for compatibility)
    if not run_command("pip install numpy>=1.19.5,<1.25.0", "Installing NumPy"):
        logger.error("Failed to install NumPy. Exiting.")
        return False
    
    # Install SciPy (compatible with NumPy)
    if not run_command("pip install scipy>=1.7.0,<1.11.0", "Installing SciPy"):
        logger.warning("Failed to install SciPy. Continuing anyway.")
    
    # Install other dependencies from requirements.txt
    if not run_command("pip install -r requirements.txt", "Installing dependencies from requirements.txt"):
        logger.warning("Some dependencies failed to install. Continuing anyway.")
    
    # Step 3: Install FAISS separately (it can be tricky)
    logger.info("Installing FAISS...")
    if not run_command("pip install faiss-cpu==1.7.2", "Installing FAISS"):
        logger.warning("Failed to install FAISS. Trying alternative method...")
        
        # Try alternative installation method
        if not run_command("pip install faiss-cpu", "Installing FAISS (latest version)"):
            logger.error("Failed to install FAISS. You may need to install it manually.")
    
    # Step 4: Install Sentence Transformers
    logger.info("Installing Sentence Transformers...")
    if not run_command("pip install sentence-transformers==2.2.2", "Installing Sentence Transformers"):
        logger.error("Failed to install Sentence Transformers. Exiting.")
        return False
    
    # Step 5: Install SciSpacy if requested
    if args.scispacy:
        logger.info("Installing SciSpacy from GitHub...")
        
        # Clone the repository if it doesn't exist
        if not os.path.exists("scispacy"):
            if not run_command("git clone https://github.com/allenai/scispacy.git", "Cloning SciSpacy repository"):
                logger.error("Failed to clone SciSpacy repository. Exiting.")
                return False
        
        # Modify setup.py to remove nmslib dependency
        setup_py_path = os.path.join("scispacy", "setup.py")
        if os.path.exists(setup_py_path):
            try:
                with open(setup_py_path, 'r') as f:
                    setup_py = f.read()
                
                # Remove nmslib dependency
                if "nmslib" in setup_py:
                    setup_py = setup_py.replace("'nmslib>=1.7.3.6',", "")
                    setup_py = setup_py.replace("'nmslib>=1.7.3.6'", "")
                    
                    with open(setup_py_path, 'w') as f:
                        f.write(setup_py)
                    
                    logger.info("Removed nmslib dependency from setup.py")
                else:
                    logger.info("nmslib dependency not found in setup.py")
            except Exception as e:
                logger.error(f"Error modifying setup.py: {e}")
                return False
        
        # Install SciSpacy
        os.chdir("scispacy")
        if not run_command("pip install .", "Installing SciSpacy"):
            logger.error("Failed to install SciSpacy. Exiting.")
            os.chdir("..")
            return False
        os.chdir("..")
    
    # Step 6: Install the scientific spaCy model
    logger.info("Installing scientific spaCy model...")
    if not run_command(
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz",
        "Installing en_core_sci_sm"
    ):
        logger.error("Failed to install scientific spaCy model. Exiting.")
        return False
    
    # Step 7: Test the installation
    logger.info("Testing installation...")
    run_command("python test_installation.py", "Running installation tests")
    
    logger.info("Environment setup completed!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Set up the environment for the Medical Q&A Chatbot')
    parser.add_argument('--clean', action='store_true', help='Uninstall existing packages before installation')
    parser.add_argument('--scispacy', action='store_true', help='Install SciSpacy from GitHub')
    args = parser.parse_args()
    
    if setup_environment(args):
        logger.info("Environment setup completed successfully!")
        return 0
    else:
        logger.error("Environment setup failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
