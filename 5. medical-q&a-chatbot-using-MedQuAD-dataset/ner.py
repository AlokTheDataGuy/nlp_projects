import spacy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalNER:
    def __init__(self, model_name="en_core_sci_sm"):
        """
        Initialize the Medical Named Entity Recognition module

        Args:
            model_name: Name of the spaCy model to use
        """
        logger.info(f"Loading SciSpacy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
            logger.info("SciSpacy model loaded successfully")

            # Try to add abbreviation detector if available
            try:
                if "abbreviation_detector" not in self.nlp.pipe_names:
                    from scispacy.abbreviation import AbbreviationDetector
                    self.nlp.add_pipe("abbreviation_detector")
                    logger.info("Added abbreviation detector to pipeline")
            except Exception as e:
                logger.warning(f"Could not add abbreviation detector: {e}")
                logger.warning("Continuing without abbreviation detection")

        except Exception as e:
            logger.error(f"Error loading SciSpacy model: {e}")
            raise

    def extract_entities(self, text):
        """
        Extract medical entities from text

        Args:
            text: Input text to process

        Returns:
            Dictionary containing entities, semantic types, and CUIs
        """
        doc = self.nlp(text)

        # Extract entities
        entities = [ent.text for ent in doc.ents]

        # Extract abbreviations if available
        abbreviations = []
        if hasattr(doc._, 'abbreviations') and doc._.abbreviations:
            for abrv in doc._.abbreviations:
                abbreviations.append({
                    'abrv': abrv.text,
                    'long_form': abrv._.long_form.text
                })

        # Get entity labels
        entity_labels = [ent.label_ for ent in doc.ents]

        return {
            "entities": entities,
            "entity_labels": entity_labels,
            "abbreviations": abbreviations,
            "semantic_types": [],  # Placeholder for compatibility
            "cuis": []  # Placeholder for compatibility
        }
