import React, { useState } from 'react';
import ConceptGraph from './ConceptGraph';
import PaperNetwork from './PaperNetwork';

/**
 * Container for visualization components with tabs
 */
const VisualizationContainer = ({ concepts = [], papers = [], relationships = [] }) => {
  const [activeTab, setActiveTab] = useState('concepts');
  
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden h-full flex flex-col">
      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        <button
          className={`py-3 px-4 font-medium text-sm focus:outline-none ${
            activeTab === 'concepts'
              ? 'text-blue-400 border-b-2 border-blue-400'
              : 'text-gray-400 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('concepts')}
        >
          Concept Graph
        </button>
        <button
          className={`py-3 px-4 font-medium text-sm focus:outline-none ${
            activeTab === 'papers'
              ? 'text-blue-400 border-b-2 border-blue-400'
              : 'text-gray-400 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('papers')}
        >
          Paper Network
        </button>
      </div>
      
      {/* Visualization content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'concepts' ? (
          <ConceptGraph concepts={concepts} links={generateConceptLinks(concepts)} />
        ) : (
          <PaperNetwork papers={papers} relationships={relationships} />
        )}
      </div>
    </div>
  );
};

/**
 * Generate sample links between concepts
 */
const generateConceptLinks = (concepts) => {
  const links = [];
  
  if (!concepts || concepts.length < 2) return links;
  
  // Create some sample links
  for (let i = 0; i < concepts.length; i++) {
    for (let j = i + 1; j < concepts.length; j++) {
      // Only create links between some concepts
      if (Math.random() > 0.5) continue;
      
      links.push({
        source: concepts[i].name,
        target: concepts[j].name,
        strength: Math.random() * 0.5 + 0.1
      });
    }
  }
  
  return links;
};

export default VisualizationContainer;
