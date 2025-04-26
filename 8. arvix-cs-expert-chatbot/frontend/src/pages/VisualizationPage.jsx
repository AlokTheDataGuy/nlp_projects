import React, { useState, useEffect } from 'react';
import Layout from '../components/Layout/Layout';
import VisualizationContainer from '../components/Visualization/VisualizationContainer';
import { useChat } from '../hooks/useChat';
import { usePaperSearch } from '../hooks/usePaperSearch';

/**
 * Concept visualization page
 */
const VisualizationPage = () => {
  const { messages } = useChat();
  const { papers } = usePaperSearch();
  const [concepts, setConcepts] = useState([]);
  const [relationships, setRelationships] = useState([]);
  
  // Extract concepts from messages
  useEffect(() => {
    if (!messages || messages.length === 0) return;
    
    const extractedConcepts = new Set();
    
    messages.forEach(message => {
      if (message.role === 'assistant' && message.content) {
        // Simple extraction - split by spaces and filter
        const words = message.content.split(/\s+/);
        const filteredWords = words.filter(word => {
          // Keep words that are likely to be concepts (capitalized, longer than 3 chars)
          return word.length > 3 && 
                 /^[A-Z][a-z]+/.test(word) && 
                 !['This', 'That', 'These', 'Those', 'There', 'Their', 'They'].includes(word);
        });
        
        filteredWords.forEach(word => {
          const cleanWord = word.replace(/[.,;:!?()]/g, '');
          extractedConcepts.add(cleanWord);
        });
      }
    });
    
    // Convert to array of objects with random weights and categories
    const conceptsArray = Array.from(extractedConcepts).map(name => ({
      name,
      weight: Math.random() * 0.5 + 0.5,
      category: ['AI', 'ML', 'NLP', 'CV', 'SE', 'DB', 'Theory', 'Systems'][
        Math.floor(Math.random() * 8)
      ]
    }));
    
    setConcepts(conceptsArray);
  }, [messages]);
  
  // Generate relationships between papers
  useEffect(() => {
    if (!papers || papers.length < 2) {
      setRelationships([]);
      return;
    }
    
    const rels = [];
    
    // Create some sample relationships
    for (let i = 0; i < papers.length; i++) {
      for (let j = i + 1; j < papers.length; j++) {
        // Only create relationships between some papers
        if (Math.random() > 0.7) continue;
        
        const types = ['cites', 'similar', 'builds-on'];
        const type = types[Math.floor(Math.random() * types.length)];
        
        rels.push({
          source: papers[i].id,
          target: papers[j].id,
          type
        });
      }
    }
    
    setRelationships(rels);
  }, [papers]);
  
  return (
    <Layout>
      <div className="h-full p-4 flex flex-col">
        <div className="mb-4">
          <h1 className="text-2xl font-bold text-white mb-2">Concept Visualization</h1>
          <p className="text-gray-400">
            Explore relationships between computer science concepts and research papers.
          </p>
        </div>
        
        <div className="flex-1">
          <VisualizationContainer 
            concepts={concepts} 
            papers={papers} 
            relationships={relationships} 
          />
        </div>
      </div>
    </Layout>
  );
};

export default VisualizationPage;
