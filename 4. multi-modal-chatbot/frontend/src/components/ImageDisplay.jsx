import { useState } from 'react';
import './ImageDisplay.css';

const ImageDisplay = ({ src, alt }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };
  
  return (
    <>
      <img 
        src={src} 
        alt={alt} 
        className="message-image" 
        onClick={toggleExpand}
      />
      
      {isExpanded && (
        <div className="image-modal" onClick={toggleExpand}>
          <div className="image-modal-content">
            <img src={src} alt={alt} />
            <button className="close-modal" onClick={toggleExpand}>×</button>
          </div>
        </div>
      )}
    </>
  );
};

export default ImageDisplay;
