import React, { ReactElement } from 'react';
import { Message, Entity } from '../types/index';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const { text, sender, isWelcome, isConversational, isUserMessage, isLoading, entities, source } = message;

  // Function to format text with markdown-like syntax
  const formatText = (text: string, entities?: Entity[]) => {
    // We'll handle entities separately in the rendering process
    let processedText = text;

    // Process headings, bullet points, and paragraphs
    const lines = processedText.split('\n');

    // Group bullet points under their headings
    const formattedContent: ReactElement[] = [];
    let currentList: ReactElement[] = [];
    let inList = false;

    lines.forEach((line, index) => {
      const trimmedLine = line.trim();

      // Main heading (all caps or first line)
      if (index === 0 || trimmedLine.match(/^[A-Z][A-Z\s]+$/)) {
        if (inList) {
          formattedContent.push(<ul key={`list-${index}`} className="list-disc pl-5 mb-3">{currentList}</ul>);
          currentList = [];
          inList = false;
        }
        formattedContent.push(
          <h2 key={`main-${index}`} className="font-bold text-gray-900 text-lg mb-2">
            {trimmedLine}
          </h2>
        );
        return;
      }

      // Subheadings (ending with colon)
      if (trimmedLine.match(/^[A-Za-z\s]+:$/)) {
        if (inList) {
          formattedContent.push(<ul key={`list-${index}`} className="list-disc pl-5 mb-3">{currentList}</ul>);
          currentList = [];
          inList = false;
        }
        formattedContent.push(
          <h3 key={`heading-${index}`} className="font-semibold text-gray-800 mt-3 mb-1">
            {trimmedLine}
          </h3>
        );
        return;
      }

      // Bullet points
      if (trimmedLine.startsWith('•') || trimmedLine.startsWith('-') || trimmedLine.startsWith('*')) {
        const bulletContent = trimmedLine.replace(/^[•\-*]\s*/, '');
        inList = true;

        // Handle entities in bullet points if they exist
        if (entities && entities.length > 0) {
          // Find entities that are in this bullet point
          const bulletEntities = entities.filter(entity => {
            // Check if the entity is within this bullet point
            return text.indexOf(bulletContent) <= entity.end &&
                   text.indexOf(bulletContent) + bulletContent.length >= entity.start;
          });

          if (bulletEntities.length > 0) {
            // Create a bullet point with highlighted entities
            const parts: ReactElement[] = [];
            let lastIndex = 0;

            // Sort entities by start position
            const sortedEntities = [...bulletEntities].sort((a, b) => a.start - b.start);

            sortedEntities.forEach((entity, i) => {
              // Adjust entity positions relative to this bullet point
              const bulletStart = text.indexOf(bulletContent);
              const relativeStart = Math.max(0, entity.start - bulletStart);
              const relativeEnd = Math.min(bulletContent.length, entity.end - bulletStart);

              // Add text before entity
              if (relativeStart > lastIndex) {
                parts.push(
                  <span key={`text-${i}`}>{bulletContent.substring(lastIndex, relativeStart)}</span>
                );
              }

              // Add highlighted entity
              parts.push(
                <span
                  key={`entity-${i}`}
                  className="bg-yellow-100 text-yellow-800 px-1 rounded"
                  title={`Type: ${entity.type}`}
                >
                  {bulletContent.substring(relativeStart, relativeEnd)}
                </span>
              );

              lastIndex = relativeEnd;
            });

            // Add any remaining text
            if (lastIndex < bulletContent.length) {
              parts.push(
                <span key="text-end">{bulletContent.substring(lastIndex)}</span>
              );
            }

            currentList.push(<li key={`bullet-${index}`} className="mb-1">{parts}</li>);
          } else {
            // No entities in this bullet point, just add the text
            currentList.push(<li key={`bullet-${index}`} className="mb-1">{bulletContent}</li>);
          }
        } else {
          // No entities at all, just add the text
          currentList.push(<li key={`bullet-${index}`} className="mb-1">{bulletContent}</li>);
        }

        return;
      }

      // Regular paragraph
      if (trimmedLine) {
        if (inList) {
          formattedContent.push(<ul key={`list-${index}`} className="list-disc pl-5 mb-3">{currentList}</ul>);
          currentList = [];
          inList = false;
        }

        // Handle entities in this paragraph if they exist
        if (entities && entities.length > 0) {
          // Find entities that are in this line
          const lineEntities = entities.filter(entity => {
            // Check if the entity is within this line
            return text.indexOf(trimmedLine) <= entity.end &&
                   text.indexOf(trimmedLine) + trimmedLine.length >= entity.start;
          });

          if (lineEntities.length > 0) {
            // Create a paragraph with highlighted entities
            const parts: ReactElement[] = [];
            let lastIndex = 0;

            // Sort entities by start position
            const sortedEntities = [...lineEntities].sort((a, b) => a.start - b.start);

            sortedEntities.forEach((entity, i) => {
              // Adjust entity positions relative to this line
              const lineStart = text.indexOf(trimmedLine);
              const relativeStart = Math.max(0, entity.start - lineStart);
              const relativeEnd = Math.min(trimmedLine.length, entity.end - lineStart);

              // Add text before entity
              if (relativeStart > lastIndex) {
                parts.push(
                  <span key={`text-${i}`}>{trimmedLine.substring(lastIndex, relativeStart)}</span>
                );
              }

              // Add highlighted entity
              parts.push(
                <span
                  key={`entity-${i}`}
                  className="bg-yellow-100 text-yellow-800 px-1 rounded"
                  title={`Type: ${entity.type}`}
                >
                  {trimmedLine.substring(relativeStart, relativeEnd)}
                </span>
              );

              lastIndex = relativeEnd;
            });

            // Add any remaining text
            if (lastIndex < trimmedLine.length) {
              parts.push(
                <span key="text-end">{trimmedLine.substring(lastIndex)}</span>
              );
            }

            formattedContent.push(<p key={`para-${index}`} className="mb-2">{parts}</p>);
          } else {
            // No entities in this line, just add the text
            formattedContent.push(<p key={`para-${index}`} className="mb-2">{trimmedLine}</p>);
          }
        } else {
          // No entities at all, just add the text
          formattedContent.push(<p key={`para-${index}`} className="mb-2">{trimmedLine}</p>);
        }

        return;
      }

      // Empty line - ignore if we're in a list
      if (!inList && trimmedLine === '') {
        formattedContent.push(<div key={`space-${index}`} className="h-2"></div>);
      }
    });

    // Add any remaining list items
    if (inList && currentList.length > 0) {
      formattedContent.push(<ul key="list-final" className="list-disc pl-5 mb-3">{currentList}</ul>);
    }

    return <div className="formatted-text">{formattedContent}</div>;
  };

  // Handle special message types
  if (isWelcome) {
    return (
      <div className="my-4 p-4 bg-blue-100 text-blue-800 rounded-lg text-center mx-auto" style={{ maxWidth: '90%' }}>
        <div className="flex items-center justify-center mb-2">
          <svg className="w-6 h-6 mr-2 text-blue-600" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17 12h-5v5h5v-5zM16 1v2H8V1H6v2H5c-1.11 0-1.99.9-1.99 2L3 19c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2h-1V1h-2zm3 18H5V8h14v11z"/>
          </svg>
          <span className="font-bold">Medical Assistant</span>
        </div>
        {text}
      </div>
    );
  }

  // Handle loading messages
  if (isLoading) {
    return (
      <div className="my-5 flex justify-start">
        <div className="w-8 h-8 mr-2 flex-shrink-0 text-gray-500">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
        </div>
        <div className="max-w-[75%] p-4 rounded-lg bg-blue-50 text-blue-800 rounded-tl-none flex items-center border border-blue-100">
          <div className="flex items-center">
            <svg className="animate-spin h-5 w-5 mr-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="font-medium">{text}</span>
          </div>
        </div>
      </div>
    );
  }

  // Handle user messages
  if (isUserMessage) {
    return (
      <div className="my-5 flex justify-end">
        <div className="max-w-[75%] p-3 rounded-lg bg-blue-500 text-white rounded-tr-none">
          <p>{text}</p>
        </div>
        <div className="w-8 h-8 ml-2 flex-shrink-0 text-blue-500">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0-6c1.1 0 2 .9 2 2s-.9 2-2 2-2-.9-2-2 .9-2 2-2zm0 7c-2.67 0-8 1.34-8 4v3h16v-3c0-2.66-5.33-4-8-4zm6 5H6v-.99c.2-.72 3.3-2.01 6-2.01s5.8 1.29 6 2v1z"/>
          </svg>
        </div>
      </div>
    );
  }

  // Handle conversational messages
  if (isConversational) {
    return (
      <div className="my-5 flex justify-start">
        <div className="w-8 h-8 mr-2 flex-shrink-0 text-blue-500">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 9h-2V5h2v6zm0 4h-2v-2h2v2z"/>
          </svg>
        </div>
        <div className="max-w-[75%] p-3 rounded-lg bg-blue-100 text-blue-800 rounded-tl-none">
          <p>{text}</p>
        </div>
      </div>
    );
  }

  // For bot messages with medical information
  return (
    <div className="my-5 flex justify-start">
      <div className="w-8 h-8 mr-2 flex-shrink-0 text-gray-500">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
      </div>
      <div className="max-w-[75%] p-3 rounded-lg bg-gray-200 text-gray-800 rounded-tl-none">
        <div className="formatted-message">
          {formatText(text, entities)}
        </div>
        {source && source.includes('Llama 3.1') && (
          <div className="mt-1 text-xs flex items-center">
            <span className="bg-green-100 text-green-800 text-xs px-1.5 py-0.5 rounded flex items-center">
              <svg className="w-3 h-3 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
              </svg>
              Enhanced by Llama 3.1
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
