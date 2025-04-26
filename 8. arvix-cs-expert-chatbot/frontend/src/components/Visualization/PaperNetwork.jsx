import React, { useEffect, useRef } from 'react';

/**
 * Paper network visualization component
 * This is a placeholder that would be implemented with a visualization library like D3.js
 */
const PaperNetwork = ({ papers, relationships }) => {
  const canvasRef = useRef(null);
  
  // Simulate a force-directed graph with canvas
  useEffect(() => {
    if (!canvasRef.current || !papers || papers.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Set up the simulation
    const nodes = papers.map((paper, i) => ({
      id: paper.id,
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 15,
      color: 'rgba(59, 130, 246, 0.7)', // blue
      label: paper.title.length > 20 ? paper.title.substring(0, 20) + '...' : paper.title
    }));
    
    // Animation loop
    let animationFrameId;
    
    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Draw relationships
      if (relationships && relationships.length > 0) {
        relationships.forEach(rel => {
          const source = nodes.find(n => n.id === rel.source);
          const target = nodes.find(n => n.id === rel.target);
          
          if (source && target) {
            // Different line styles based on relationship type
            switch (rel.type) {
              case 'cites':
                ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)'; // green
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                break;
              case 'similar':
                ctx.strokeStyle = 'rgba(139, 92, 246, 0.5)'; // purple
                ctx.lineWidth = 1;
                ctx.setLineDash([]);
                break;
              case 'builds-on':
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)'; // red
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                break;
              default:
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.lineWidth = 1;
                ctx.setLineDash([]);
            }
            
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);
            ctx.stroke();
            
            // Draw arrow for directed relationships
            if (rel.type === 'cites' || rel.type === 'builds-on') {
              const dx = target.x - source.x;
              const dy = target.y - source.y;
              const angle = Math.atan2(dy, dx);
              
              const arrowLength = 10;
              const arrowWidth = 5;
              
              // Calculate position for arrow (near target)
              const arrowX = target.x - Math.cos(angle) * target.radius;
              const arrowY = target.y - Math.sin(angle) * target.radius;
              
              ctx.beginPath();
              ctx.moveTo(arrowX, arrowY);
              ctx.lineTo(
                arrowX - arrowLength * Math.cos(angle) + arrowWidth * Math.sin(angle),
                arrowY - arrowLength * Math.sin(angle) - arrowWidth * Math.cos(angle)
              );
              ctx.lineTo(
                arrowX - arrowLength * Math.cos(angle) - arrowWidth * Math.sin(angle),
                arrowY - arrowLength * Math.sin(angle) + arrowWidth * Math.cos(angle)
              );
              ctx.closePath();
              ctx.fillStyle = ctx.strokeStyle;
              ctx.fill();
            }
          }
        });
        
        // Reset line dash
        ctx.setLineDash([]);
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
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.label, node.x, node.y + node.radius + 12);
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
  }, [papers, relationships]);
  
  // If no papers, show placeholder
  if (!papers || papers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 text-gray-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3 className="text-xl font-medium text-gray-300 mb-2">No Papers to Visualize</h3>
        <p className="text-gray-500 max-w-md">
          Search for papers or chat with the AI assistant to discover related papers.
          The visualization will show how papers are connected through citations and topics.
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
        <h4 className="text-sm font-medium text-white mb-2">Relationship Types</h4>
        <div className="space-y-2">
          <div className="flex items-center text-xs">
            <div className="w-4 h-0.5 bg-green-500 mr-2"></div>
            <span className="text-gray-300">Cites</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-4 h-0.5 bg-purple-500 mr-2"></div>
            <span className="text-gray-300">Similar</span>
          </div>
          <div className="flex items-center text-xs">
            <div className="w-4 h-0.5 bg-red-500 mr-2 border-t border-dashed"></div>
            <span className="text-gray-300">Builds On</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PaperNetwork;
