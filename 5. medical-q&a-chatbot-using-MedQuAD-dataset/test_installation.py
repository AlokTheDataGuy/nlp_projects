import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_spacy():
    """Test spaCy and SciSpacy installation"""
    try:
        import spacy
        logger.info(f"spaCy version: {spacy.__version__}")

        # Test loading the scientific model
        try:
            nlp = spacy.load("en_core_sci_sm")
            logger.info("Successfully loaded en_core_sci_sm model")

            # Test basic NER
            text = "The patient was diagnosed with diabetes mellitus and hypertension."
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents]
            logger.info(f"Extracted entities: {entities}")

            # Test if scispacy is installed
            try:
                import scispacy
                logger.info(f"SciSpacy version: {scispacy.__version__}")

                # Test if entity linker is available
                try:
                    # Different ways to check for pipe availability depending on spaCy version
                    if hasattr(spacy.registry.factories, 'get_names'):
                        pipe_names = spacy.registry.factories.get_names("pipe")
                        has_linker = "scispacy_linker" in pipe_names
                    else:
                        # For older spaCy versions
                        has_linker = "scispacy_linker" in nlp.factory_names

                    if has_linker:
                        logger.info("SciSpacy linker is available")
                    else:
                        logger.info("SciSpacy linker is not available in registry")
                except Exception as e:
                    logger.warning(f"Error checking for scispacy_linker: {e}")
                    has_linker = False

                    # Try to add the linker
                    if "entity_linker" not in nlp.pipe_names:
                        try:
                            nlp.add_pipe("scispacy_linker",
                                        config={"resolve_abbreviations": True,
                                                "linker_name": "umls"})
                            logger.info("Successfully added UMLS entity linker")

                            # Test entity linking
                            doc = nlp(text)
                            for ent in doc.ents:
                                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                                    logger.info(f"Entity '{ent.text}' linked to: {ent._.kb_ents[0][0]}")
                        except Exception as e:
                            logger.warning(f"Could not add UMLS entity linker: {e}")
                else:
                    logger.warning("SciSpacy linker is not available")
            except ImportError:
                logger.warning("SciSpacy is not installed")
        except Exception as e:
            logger.error(f"Error loading en_core_sci_sm model: {e}")
            return False
    except ImportError:
        logger.error("spaCy is not installed")
        return False

    return True

def test_sentence_transformers():
    """Test Sentence Transformers installation"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Sentence Transformers is installed")

            # Test loading a model
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Successfully loaded Sentence Transformers model")

                # Test embedding
                text = "What are the symptoms of diabetes?"
                embedding = model.encode(text)
                logger.info(f"Generated embedding with shape: {embedding.shape}")
            except Exception as e:
                logger.error(f"Error loading Sentence Transformers model: {e}")
                logger.warning("Try installing with: pip install sentence-transformers")
                return False
        except ImportError:
            logger.error("Sentence Transformers is not installed")
            logger.warning("Install with: pip install sentence-transformers")
            return False
    except ImportError:
        logger.warning("PyTorch is not installed, which is required for Sentence Transformers")
        logger.warning("Install with: pip install torch")
        return False

    return True

def test_numpy_scipy_compatibility():
    """Test NumPy and SciPy compatibility"""
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")

        try:
            import scipy
            logger.info(f"SciPy version: {scipy.__version__}")

            # Check version compatibility
            np_version = tuple(map(int, np.__version__.split('.')[:2]))
            if np_version >= (1, 25):
                logger.warning("NumPy version is >= 1.25.0, which may cause compatibility issues with SciPy")
                logger.warning("Consider downgrading NumPy: pip install numpy>=1.19.5,<1.25.0")

            # Test basic operations to check compatibility
            try:
                # Create a simple array and perform operations
                arr = np.array([[1, 2], [3, 4]])
                result = scipy.linalg.det(arr)
                logger.info(f"NumPy/SciPy compatibility test passed: det([[1, 2], [3, 4]]) = {result}")
                return True
            except Exception as e:
                logger.error(f"NumPy/SciPy compatibility test failed: {e}")
                logger.warning("Try reinstalling compatible versions:")
                logger.warning("pip uninstall -y numpy scipy")
                logger.warning("pip install numpy>=1.19.5,<1.25.0")
                logger.warning("pip install scipy>=1.7.0,<1.11.0")
                return False
        except ImportError:
            logger.warning("SciPy is not installed")
            return True  # Not critical for this test
    except ImportError:
        logger.error("NumPy is not installed")
        return False

def test_faiss():
    """Test FAISS installation"""
    try:
        import faiss
        import numpy as np
        logger.info(f"FAISS is installed (version: {getattr(faiss, '__version__', 'unknown')})")

        # Test creating a simple index
        try:
            dimension = 128
            index = faiss.IndexFlatL2(dimension)

            # Add some vectors
            vectors = np.random.random((10, dimension)).astype('float32')
            index.add(vectors)

            # Test search
            query = np.random.random((1, dimension)).astype('float32')
            distances, indices = index.search(query, 3)

            logger.info(f"FAISS search returned indices: {indices}")
            return True
        except Exception as e:
            logger.error(f"Error testing FAISS: {e}")
            logger.warning("Try reinstalling FAISS: pip install faiss-cpu")
            return False
    except ImportError as e:
        logger.error(f"FAISS is not installed: {e}")
        logger.warning("Install FAISS with: pip install faiss-cpu")
        return False

def test_flask():
    """Test Flask installation"""
    try:
        import flask
        logger.info(f"Flask version: {flask.__version__}")

        # Test werkzeug version
        try:
            import werkzeug
            logger.info(f"Werkzeug version: {werkzeug.__version__}")

            # Check for url_quote issue
            try:
                from werkzeug.urls import url_quote
                logger.info("Werkzeug url_quote is available")
            except ImportError:
                logger.warning("Werkzeug url_quote is not available. You may need to downgrade werkzeug to version 2.0.3")
                logger.warning("Run: pip uninstall werkzeug && pip install werkzeug==2.0.3")
        except ImportError:
            logger.warning("Werkzeug is not installed")

        # Test flask-cors
        try:
            from flask_cors import CORS
            logger.info("Flask-CORS is installed")
        except ImportError as e:
            logger.warning(f"Flask-CORS is not installed: {e}")

        logger.info("Flask is installed")
        return True
    except ImportError as e:
        logger.error(f"Error importing Flask: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Testing installation...")

    tests = [
        ("NumPy and SciPy compatibility", test_numpy_scipy_compatibility, True),  # Critical
        ("spaCy and SciSpacy", test_spacy, True),  # Critical
        ("Sentence Transformers", test_sentence_transformers, True),  # Critical
        ("FAISS", test_faiss, True),  # Critical
        ("Flask", test_flask, False)  # Non-critical
    ]

    all_passed = True
    critical_failed = False

    for name, test_func, is_critical in tests:
        logger.info(f"\nTesting {name}...")
        try:
            if test_func():
                logger.info(f"✅ {name} test passed")
            else:
                if is_critical:
                    logger.error(f"❌ {name} test failed (CRITICAL)")
                    critical_failed = True
                else:
                    logger.warning(f"⚠️ {name} test failed (NON-CRITICAL)")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ Error during {name} test: {e}")
            if is_critical:
                critical_failed = True
            all_passed = False

    if all_passed:
        logger.info("\n✅ All tests passed! You're ready to run the application.")
        return 0
    elif critical_failed:
        logger.error("\n❌ Some critical tests failed. Please fix the issues before running the application.")
        return 1
    else:
        logger.warning("\n⚠️ Some non-critical tests failed. You may still be able to run the application, but some features might not work properly.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
