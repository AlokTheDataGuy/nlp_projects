import { useState, useRef } from 'react';
import ImageUpload from './ImageUpload';
import { sendMessage } from '../services/api';
import './ChatInterface.css';

const ChatInterface = ({ onSendMessage, setLoading, messages }) => {
  const [message, setMessage] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!message.trim() && !image) return;

    // Store the current message and image for API call
    const currentMessage = message;
    const currentImage = image;
    const currentImagePreview = imagePreview;

    // Clear inputs immediately
    setMessage('');
    setImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // Add user message to the chat
    onSendMessage({
      id: Date.now(),
      role: 'user',
      content: currentMessage,
      image: currentImagePreview
    });

    // Prepare for API call
    setLoading(true);

    try {
      // Create conversation history for the API
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Send message to API
      const response = await sendMessage(currentMessage, currentImage, history);

      // Add bot response to chat
      onSendMessage({
        id: Date.now() + 1,
        role: 'assistant',
        content: response.text
      });
    } catch (error) {
      console.error('Error sending message:', error);
      onSendMessage({
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.',
        error: true
      });
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (file) => {
    setImage(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="chat-interface">
      {imagePreview && (
        <div className="image-preview-container">
          <img src={imagePreview} alt="Preview" className="image-preview" />
          <button
            className="remove-image-btn"
            onClick={handleRemoveImage}
            aria-label="Remove image"
          >
            ×
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="chat-form">
        <ImageUpload
          onImageUpload={handleImageUpload}
          ref={fileInputRef}
        />

        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
          className="message-input"
        />

        <button
          type="submit"
          className="send-button"
          disabled={!message.trim() && !image}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
