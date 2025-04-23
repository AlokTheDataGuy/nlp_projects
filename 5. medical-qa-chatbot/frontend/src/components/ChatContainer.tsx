import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { Message } from '../types/index';
import { askQuestion } from '../services/api';
import { curateWithLlama } from '../services/llmService';

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      text: 'Welcome to the Medical Q&A Chatbot! I use both the MedQuAD dataset and Meditron LLM to answer your medical questions. Ask me anything about medical conditions, symptoms, treatments, and more.',
      sender: 'bot',
      timestamp: new Date(),
      isWelcome: true,
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isCurating, setIsCurating] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    // Add user message
    const userMessage: Message = {
      id: uuidv4(),
      text,
      sender: 'user',
      timestamp: new Date(),
      isUserMessage: true,  // Flag to identify user messages
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Add a loading message that will be replaced when the real response arrives
    const loadingMessageId = uuidv4();
    const loadingMessage: Message = {
      id: loadingMessageId,
      text: "Wait for response...",
      sender: 'bot',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages((prev) => [...prev, loadingMessage]);

    try {
      // Call API
      const response = await askQuestion({ question: text });

      setIsLoading(false);

      // Skip curation for conversational messages
      const isConversational = response.answers.length === 1 && response.answers[0].source === 'Conversation';
      if (!isConversational && response.answers.length > 0) {
        setIsCurating(true);

        // Add a loading message for the curation phase
        const curationLoadingId = uuidv4();
        const curationLoadingMessage: Message = {
          id: curationLoadingId,
          text: "Enhancing response with Llama 3.1...",
          sender: 'bot',
          timestamp: new Date(),
          isLoading: true
        };

        setMessages((prev) => [...prev, curationLoadingMessage]);
      }

      // Remove the loading message
      setMessages((prev) => prev.filter(msg => !msg.isLoading));

      // Process each answer through Llama 3.1 for curation
      for (let index = 0; index < response.answers.length; index++) {
        const answer = response.answers[index];

        // Skip Llama curation for conversational responses
        if (answer.source === 'Conversation') {
          const botMessage: Message = {
            id: uuidv4(),
            text: answer.answer,
            sender: 'bot',
            timestamp: new Date(),
            entities: answer.entities,
            source: 'Conversation',
            confidence: answer.confidence,
            isConversational: true
          };

          setMessages((prev) => [...prev, botMessage]);
          setIsCurating(false);
          continue;
        }

        try {
          // Curate the answer with Llama 3.1
          const curatedAnswer = await curateWithLlama(text, answer.answer);

          const botMessage: Message = {
            id: uuidv4(),
            text: curatedAnswer,
            sender: 'bot',
            timestamp: new Date(),
            entities: answer.entities,
            source: answer.source + ' (Enhanced by Llama 3.1)',
            confidence: answer.confidence,
          };

          // Add a slight delay between multiple answers for better UX
          setMessages((prev) => [...prev, botMessage]);

          // Add a small delay between processing multiple answers
          if (index < response.answers.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 300));
          }
        } catch (error) {
          console.error('Error during curation:', error);
          // Fallback to original answer if curation fails
          const botMessage: Message = {
            id: uuidv4(),
            text: answer.answer,
            sender: 'bot',
            timestamp: new Date(),
            entities: answer.entities,
            source: answer.source,
            confidence: answer.confidence,
          };

          setMessages((prev) => [...prev, botMessage]);
        }
      }

      // If no answers, add a fallback message
      if (response.answers.length === 0) {
        // Remove the loading message if it's still there
        setMessages((prev) => prev.filter(msg => !msg.isLoading));
        const fallbackMessage: Message = {
          id: uuidv4(),
          text: "I'm sorry, I couldn't find a specific answer to your question. Please try rephrasing or ask another medical question.",
          sender: 'bot',
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, fallbackMessage]);
      }
    } catch (error) {
      console.error('Error getting answer:', error);

      // Remove the loading message
      setMessages((prev) => prev.filter(msg => !msg.isLoading));

      // Add error message
      const errorMessage: Message = {
        id: uuidv4(),
        text: 'Sorry, there was an error processing your question. Please try again later.',
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsCurating(false);
    }
  };

  return (
    <div className="max-w-6xl w-full mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-5 rounded-t-lg">
        <h1 className="text-2xl font-bold">Medical Q&A Chatbot</h1>
        <p className="text-sm">Powered by MedQuAD dataset and Meditron LLM</p>
        <div className="flex items-center mt-2">
          <span className="text-xs bg-white bg-opacity-20 px-2 py-0.5 rounded mr-2">Retrieval-based</span>
          <span className="text-xs bg-white bg-opacity-20 px-2 py-0.5 rounded">AI-enhanced</span>
        </div>
      </div>

      <div className="messages-container h-[65vh] overflow-y-auto p-4 bg-white" style={{ width: '100%' }}>
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-gray-200 p-4 bg-white">
        {isCurating && (
          <div className="mb-2 flex items-center justify-center text-sm text-green-600 bg-green-50 py-2 px-3 rounded-lg">
            <svg className="animate-spin h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>Llama 3.1 is improving the response for better understanding...</span>
          </div>
        )}
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading || isCurating} />
        <div className="text-xs text-gray-500 mt-2 flex flex-col">
          <div className="flex justify-between items-center">
            <p className="max-w-3xl">
              <strong>Disclaimer:</strong> This chatbot provides information based on the MedQuAD dataset and Meditron LLM.
              It is not a substitute for professional medical advice, diagnosis, or treatment.
            </p>
            <div className="flex items-center ml-4">
              <span className="bg-blue-100 text-blue-800 text-xs px-2 py-0.5 rounded mr-2">MedQuAD</span>
              <span className="bg-purple-100 text-purple-800 text-xs px-2 py-0.5 rounded mr-2">Meditron LLM</span>
              <span className="bg-green-100 text-green-800 text-xs px-2 py-0.5 rounded">Llama 3.1</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatContainer;
