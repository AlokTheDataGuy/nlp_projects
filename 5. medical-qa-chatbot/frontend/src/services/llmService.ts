import axios from 'axios';

interface LlamaCurationRequest {
  question: string;
  originalAnswer: string;
  model?: string;
}

interface LlamaCurationResponse {
  curatedAnswer: string;
}

// Function to curate answers using Llama 3.1
export const curateWithLlama = async (question: string, originalAnswer: string): Promise<string> => {
  try {
    // Show a message in the console for debugging
    console.log('Curating answer with Llama 3.1...');
    console.log('Original answer:', originalAnswer.substring(0, 100) + '...');

    const response = await axios.post<LlamaCurationResponse>('/api/curate', {
      question,
      originalAnswer,
      model: 'llama3.1:8b'
    });

    console.log('Curated answer:', response.data.curatedAnswer.substring(0, 100) + '...');
    return response.data.curatedAnswer;
  } catch (error) {
    console.error('Error curating with Llama:', error);
    // Return the original answer if curation fails
    return originalAnswer;
  }
};
