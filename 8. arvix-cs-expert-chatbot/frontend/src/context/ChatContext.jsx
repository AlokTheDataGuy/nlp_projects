import React, { createContext, useState, useEffect } from 'react';

// Create context
export const ChatContext = createContext();

/**
 * Chat context provider
 */
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [currentConversation, setCurrentConversation] = useState(null);
  
  // Load conversations from local storage on mount
  useEffect(() => {
    const savedConversations = localStorage.getItem('conversations');
    if (savedConversations) {
      setConversations(JSON.parse(savedConversations));
    }
    
    const currentId = localStorage.getItem('currentConversation');
    if (currentId) {
      setCurrentConversation(currentId);
      loadMessages(currentId);
    }
  }, []);
  
  // Save conversations to local storage when they change
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }
  }, [conversations]);
  
  // Save current conversation ID to local storage
  useEffect(() => {
    if (currentConversation) {
      localStorage.setItem('currentConversation', currentConversation);
    }
  }, [currentConversation]);
  
  // Load messages for a conversation
  const loadMessages = (conversationId) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation && conversation.messages) {
      setMessages(conversation.messages);
    } else {
      setMessages([]);
    }
  };
  
  // Create a new conversation
  const createConversation = (firstMessage) => {
    const id = Date.now().toString();
    const newConversation = {
      id,
      title: firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : ''),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      messages: []
    };
    
    setConversations([newConversation, ...conversations]);
    setCurrentConversation(id);
    setMessages([]);
    
    return id;
  };
  
  // Update a conversation
  const updateConversation = (id, updates) => {
    const updatedConversations = conversations.map(conversation => {
      if (conversation.id === id) {
        return { ...conversation, ...updates, updated_at: new Date().toISOString() };
      }
      return conversation;
    });
    
    setConversations(updatedConversations);
  };
  
  // Delete a conversation
  const deleteConversation = (id) => {
    const updatedConversations = conversations.filter(conversation => conversation.id !== id);
    setConversations(updatedConversations);
    
    if (currentConversation === id) {
      setCurrentConversation(updatedConversations.length > 0 ? updatedConversations[0].id : null);
      setMessages(updatedConversations.length > 0 ? updatedConversations[0].messages || [] : []);
    }
  };
  
  // Select a conversation
  const selectConversation = (id) => {
    setCurrentConversation(id);
    loadMessages(id);
  };
  
  // Clear the current chat
  const clearChat = () => {
    setCurrentConversation(null);
    setMessages([]);
    localStorage.removeItem('currentConversation');
  };
  
  // Send a message
  const sendMessage = async (content) => {
    // Create user message
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    };
    
    // Add to messages
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    
    // Create or update conversation
    let conversationId = currentConversation;
    if (!conversationId) {
      conversationId = createConversation(content);
    }
    
    // Update conversation with new messages
    updateConversation(conversationId, {
      messages: updatedMessages,
      title: conversations.find(c => c.id === conversationId)?.title || content.slice(0, 30) + (content.length > 30 ? '...' : '')
    });
    
    // Set loading state
    setLoading(true);
    
    try {
      // Simulate API call to get response
      // In a real app, this would be a fetch to your backend
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create assistant message with mock data
      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: generateMockResponse(content),
        timestamp: new Date().toISOString(),
        papers: generateMockPapers(content)
      };
      
      // Add assistant message
      const messagesWithResponse = [...updatedMessages, assistantMessage];
      setMessages(messagesWithResponse);
      
      // Update conversation with new messages
      updateConversation(conversationId, {
        messages: messagesWithResponse
      });
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
        error: true
      };
      
      const messagesWithError = [...updatedMessages, errorMessage];
      setMessages(messagesWithError);
      
      // Update conversation with error message
      updateConversation(conversationId, {
        messages: messagesWithError
      });
    } finally {
      setLoading(false);
    }
  };
  
  // Generate a mock response (for demo purposes)
  const generateMockResponse = (query) => {
    const responses = [
      `That's an interesting question about ${query}. In computer science research, this topic has been explored extensively. Recent papers have shown promising results in this area, particularly in improving efficiency and accuracy.`,
      `${query} is a fascinating area of research. Several key papers in the last few years have made significant contributions to this field. The consensus seems to be that hybrid approaches tend to yield the best results.`,
      `When it comes to ${query}, there are multiple perspectives in the research community. Some researchers focus on theoretical foundations, while others prioritize practical applications. The most cited papers tend to bridge this gap.`,
      `Research on ${query} has evolved significantly over the past decade. Early approaches relied heavily on heuristics, but modern methods leverage deep learning and other advanced techniques to achieve state-of-the-art results.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };
  
  // Generate mock papers (for demo purposes)
  const generateMockPapers = (query) => {
    const keywords = query.toLowerCase().split(' ');
    
    const paperTemplates = [
      {
        title: 'Advances in Deep Learning for Natural Language Processing',
        authors: ['Zhang, J.', 'Smith, A.', 'Johnson, R.'],
        abstract: 'This paper presents recent advances in deep learning approaches for natural language processing tasks. We review state-of-the-art models and propose new architectures that achieve superior performance on benchmark datasets.',
        published_date: '2023-03-15',
        categories: ['cs.CL', 'cs.AI', 'cs.LG'],
        keywords: ['deep learning', 'nlp', 'language', 'neural networks', 'transformers']
      },
      {
        title: 'Efficient Algorithms for Large-Scale Graph Processing',
        authors: ['Brown, K.', 'Davis, M.', 'Wilson, T.'],
        abstract: 'We introduce novel algorithms for processing large-scale graphs with billions of edges. Our approach reduces computational complexity while maintaining accuracy, enabling real-time analysis of massive network structures.',
        published_date: '2022-11-28',
        categories: ['cs.DS', 'cs.DC'],
        keywords: ['algorithms', 'graphs', 'networks', 'distributed computing', 'efficiency']
      },
      {
        title: 'Reinforcement Learning for Robotic Control Systems',
        authors: ['Lee, S.', 'Garcia, C.', 'Patel, N.'],
        abstract: 'This research explores the application of reinforcement learning techniques to robotic control systems. We demonstrate how our approach enables robots to learn complex tasks through interaction with their environment.',
        published_date: '2023-01-10',
        categories: ['cs.RO', 'cs.AI', 'cs.LG'],
        keywords: ['reinforcement learning', 'robotics', 'control systems', 'ai', 'machine learning']
      },
      {
        title: 'Privacy-Preserving Methods in Federated Learning',
        authors: ['Wang, L.', 'Kumar, A.', 'Chen, Y.'],
        abstract: 'We present novel privacy-preserving techniques for federated learning systems. Our methods enable collaborative model training while ensuring that sensitive user data remains protected from potential adversaries.',
        published_date: '2022-09-05',
        categories: ['cs.CR', 'cs.LG', 'cs.DC'],
        keywords: ['privacy', 'security', 'federated learning', 'distributed systems', 'encryption']
      },
      {
        title: 'Computer Vision Approaches for Autonomous Vehicles',
        authors: ['Miller, E.', 'Thompson, J.', 'Anderson, P.'],
        abstract: 'This paper surveys recent computer vision techniques for autonomous vehicle perception. We analyze various approaches for object detection, semantic segmentation, and depth estimation in challenging driving scenarios.',
        published_date: '2023-02-18',
        categories: ['cs.CV', 'cs.RO'],
        keywords: ['computer vision', 'autonomous vehicles', 'object detection', 'perception', 'self-driving']
      }
    ];
    
    // Filter papers that match keywords in the query
    const relevantPapers = paperTemplates.filter(paper => {
      return keywords.some(keyword => 
        paper.title.toLowerCase().includes(keyword) || 
        paper.abstract.toLowerCase().includes(keyword) ||
        paper.keywords.some(k => k.includes(keyword))
      );
    });
    
    // If no relevant papers, return a random selection
    const papers = relevantPapers.length > 0 
      ? relevantPapers 
      : paperTemplates.sort(() => 0.5 - Math.random()).slice(0, 2);
    
    // Add IDs and URLs
    return papers.map((paper, index) => ({
      ...paper,
      id: `paper-${Date.now()}-${index}`,
      arxiv_id: `2304.${10000 + Math.floor(Math.random() * 9999)}`,
      pdf_url: `https://arxiv.org/pdf/2304.${10000 + Math.floor(Math.random() * 9999)}.pdf`,
      citation_count: Math.floor(Math.random() * 500)
    }));
  };
  
  return (
    <ChatContext.Provider
      value={{
        messages,
        loading,
        conversations,
        currentConversation,
        sendMessage,
        clearChat,
        selectConversation,
        deleteConversation
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};
