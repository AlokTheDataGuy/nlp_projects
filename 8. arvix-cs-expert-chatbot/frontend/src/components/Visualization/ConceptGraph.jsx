import React, { useEffect, useRef } from 'react';

/**
 * Concept graph visualization component
 * This is a placeholder that would be implemented with a visualization library like D3.js
 */
const ConceptGraph = ({ concepts, links }) => {
  const canvasRef = useRef(null);
  
  // Simulate a force-directed graph with canvas
  useEffect(() => {
    if (!canvasRef.current || !concepts || concepts.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Set up the simulation
    const nodes = concepts.map((concept, i) => ({
      id: concept.name,
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 20 + concept.weight * 20,
      color: getColorForCategory(concept.category),
      label: concept.name
    }));
    
    // Animation loop
    let animationFrameId;
    
    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Draw links
      if (links && links.length > 0) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1;
        
        links.forEach(link => {
          const source = nodes.find(n => n.id === link.source);
          const target = nodes.find(n => n.id === link.target);
          
          if (source && target) {
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);
            ctx.stroke();
          }
        });
      }
      
      // Draw nodes
      nodes.forEach(node => {
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
        ctx.fillStyle = node.color;
        ctx.fill();
        
        // Node border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Node label
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, node.x, node.y);
      });
      
      // Simple animation - just move nodes randomly
      nodes.forEach(node => {
        node.x += (Math.random() - 0.5) * 0.5;
        node.y += (Math.random() - 0.5) * 0.5;
        
        // Keep within bounds
        node.x = Math.max(node.radius, Math.min(width - node.radius, node.x));
        node.y = Math.max(node.radius, Math.min(height - node.radius, node.y));
      });
      
      animationFrameId = window.requestAnimationFrame(render);
    };
    
    render();
    
    // Cleanup
    return () => {
      window.cancelAnimationFrame(animationFrameId);
    };
  }, [concepts, links]);
  
  // Get color based on category
  const getColorForCategory = (category) => {
    const colors = {
      'AI': 'rgba(59, 130, 246, 0.7)', // blue
      'ML': 'rgba(16, 185, 129, 0.7)', // green
      'NLP': 'rgba(139, 92, 246, 0.7)', // purple
      'CV': 'rgba(239, 68, 68, 0.7)',   // red
      'SE': 'rgba(245, 158, 11, 0.7)',  // amber
      'DB': 'rgba(14, 165, 233, 0.7)',  // sky
      'Theory': 'rgba(168, 85, 247, 0.7)', // purple
      'Systems': 'rgba(249, 115, 22, 0.7)' // orange
    };
    
    return colors[category] || 'rgba(107, 114, 128, 0.7)'; // gray default
  };
  
  // If no concepts, show placeholder
  if (!concepts || concepts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        <h3 className="text-xl font-medium text-gray-300 mb-2">No Concepts to Visualize</h3>
        <p className="text-gray-500 max-w-md">
          Chat with the AI assistant to extract concepts from your conversations.
          The more you chat, the richer your concept graph will become.
        </p>
      </div>
    );
  }
  
  return (
    <div className="relative h-full w-full">
      <canvas 
        ref={canvasRef} 
        width={800} 
        height={600} 
        className="w-full h-full bg-gray-900 rounded-lg"
      />
      
      {/* Legend */}
      <div className="absolute top-4 right-4 bg-gray-800 bg-opacity-80 p-3 rounded-md border border-gray-700">
        <h4 className="text-sm font-medium text-white mb-2">Categories</h4>
        <div className="space-y-1">
          {['AI', 'ML', 'NLP', 'CV', 'SE', 'DB', 'Theory', 'Systems'].map(category => (
            <div key={category} className="flex items-center text-xs">
              <div 
                className="w-3 h-3 rounded-full mr-2" 
                style={{ backgroundColor: getColorForCategory(category) }}
              ></div>
              <span className="text-gray-300">{category}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ConceptGraph;
