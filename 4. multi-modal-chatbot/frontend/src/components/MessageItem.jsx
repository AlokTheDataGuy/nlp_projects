import './MessageItem.css';
import ImageDisplay from './ImageDisplay';

const MessageItem = ({ message }) => {
  const { role, content, image, error } = message;
  
  return (
    <div className={`message ${role}-message ${error ? 'error-message' : ''}`}>
      <div className="message-avatar">
        {role === 'user' ? '👤' : '🤖'}
      </div>
      
      <div className="message-content">
        {image && (
          <ImageDisplay src={image} alt={`${role} uploaded image`} />
        )}
        
        <div className="message-text">
          {content}
        </div>
      </div>
    </div>
  );
};

export default MessageItem;
