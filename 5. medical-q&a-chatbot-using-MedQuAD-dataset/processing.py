import os
import concurrent.futures
from pathlib import Path
import pandas as pd
from lxml import etree
import json

def process_xml_file(file_path):
    """Process a single XML file and return structured data"""
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        
        # Extract document metadata
        doc_id = root.get('id')
        source = root.get('source')
        url = root.get('url')
        focus = root.find('Focus').text if root.find('Focus') is not None else None
        
        # Extract category and semantic information if available
        category = None
        semantic_types = []
        cuis = []
        
        focus_annotations = root.find('FocusAnnotations')
        if focus_annotations is not None:
            category_elem = focus_annotations.find('Category')
            if category_elem is not None:
                category = category_elem.text
                
            umls = focus_annotations.find('UMLS')
            if umls is not None:
                for cui in umls.findall('.//CUI'):
                    if cui.text:
                        cuis.append(cui.text)
                for sem_type in umls.findall('.//SemanticType'):
                    if sem_type.text:
                        semantic_types.append(sem_type.text)
        
        # Extract question-answer pairs
        qa_pairs = []
        for qa_pair in root.findall('.//QAPair'):
            pair_id = qa_pair.get('pid')
            question_elem = qa_pair.find('Question')
            answer_elem = qa_pair.find('Answer')
            
            if question_elem is not None:
                q_id = question_elem.get('qid')
                q_type = question_elem.get('qtype')
                question = question_elem.text
                answer = answer_elem.text if answer_elem is not None and answer_elem.text else ""
                
                qa_pairs.append({
                    'pair_id': pair_id,
                    'question_id': q_id,
                    'question_type': q_type,
                    'question': question,
                    'answer': answer
                })
        
        return {
            'doc_id': doc_id,
            'source': source,
            'url': url,
            'focus': focus,
            'category': category,
            'semantic_types': semantic_types,
            'cuis': cuis,
            'qa_pairs': qa_pairs
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory(directory):
    """Process all XML files in a directory"""
    results = []
    directory_path = Path(directory)
    files = list(directory_path.glob('*.xml'))
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in executor.map(process_xml_file, files):
            if result:
                results.append(result)
                
                # Periodically save results to avoid memory issues
                if len(results) % 100 == 0:
                    save_batch(results, directory_path.name)
                    results = []
    
    # Save any remaining results
    if results:
        save_batch(results, directory_path.name)
    
    return f"Processed {directory_path.name}"

def save_batch(batch, source_dir):
    """Save a batch of processed results"""
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{source_dir}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(batch, f)

def main():
    base_dir = 'MedQuAD'
    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    # Process each directory
    for directory in directories:
        print(f"Processing {directory}...")
        process_directory(directory)
    
    # After processing all directories, combine the results
    combine_results()

def combine_results():
    """Combine all processed JSON files into a single dataset"""
    processed_dir = Path('data/processed')
    all_files = list(processed_dir.glob('*.json'))
    
    all_data = []
    for file in all_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    # Convert to DataFrame for easier handling
    # Extract QA pairs into flat structure
    flat_data = []
    for doc in all_data:
        for qa in doc['qa_pairs']:
            flat_data.append({
                'doc_id': doc['doc_id'],
                'source': doc['source'],
                'url': doc['url'],
                'focus': doc['focus'],
                'pair_id': qa['pair_id'],
                'question_id': qa['question_id'],
                'question_type': qa['question_type'],
                'question': qa['question'],
                'answer': qa['answer']
            })
    
    df = pd.DataFrame(flat_data)
    
    # Try with a different filename if the original is locked
    try:
        output_file = 'data/processed/medquad_complete.csv'
        df.to_csv(output_file, index=False)
    except PermissionError:
        # Try with a timestamp in the filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/processed/medquad_complete_{timestamp}.csv'
        df.to_csv(output_file, index=False)
    
    print(f"Combined dataset created with {len(df)} QA pairs")

if __name__ == "__main__":
    main()
