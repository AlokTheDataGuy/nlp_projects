import React, { createContext, useState } from 'react';

// Create context
export const PaperSearchContext = createContext();

/**
 * Paper search context provider
 */
export const PaperSearchProvider = ({ children }) => {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Search for papers
  const searchPapers = async (query, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API call to search papers
      // In a real app, this would be a fetch to your backend
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate mock papers based on query
      const mockPapers = generateMockPapers(query, options);
      setPapers(mockPapers);
    } catch (err) {
      console.error('Error searching papers:', err);
      setError('Failed to search papers. Please try again.');
      setPapers([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Generate mock papers (for demo purposes)
  const generateMockPapers = (query, options) => {
    const { categories = [], dateRange = 'all', sortBy = 'relevance' } = options;
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
      },
      {
        title: 'Quantum Computing Algorithms for Optimization Problems',
        authors: ['Gupta, R.', 'Yamamoto, K.', 'Fischer, M.'],
        abstract: 'This paper explores quantum computing algorithms for solving complex optimization problems. We demonstrate quantum advantage for specific problem classes and analyze the theoretical speedup compared to classical approaches.',
        published_date: '2022-12-05',
        categories: ['quant-ph', 'cs.DS', 'cs.CC'],
        keywords: ['quantum computing', 'algorithms', 'optimization', 'complexity theory']
      },
      {
        title: 'Explainable AI: Methods and Applications',
        authors: ['Martinez, A.', 'Kim, J.', 'Patel, S.'],
        abstract: 'We present a comprehensive survey of explainable AI techniques and their applications across various domains. Our analysis covers post-hoc explanation methods, inherently interpretable models, and evaluation metrics for explainability.',
        published_date: '2023-04-20',
        categories: ['cs.AI', 'cs.LG', 'cs.HC'],
        keywords: ['explainable ai', 'interpretability', 'transparency', 'machine learning']
      },
      {
        title: 'Secure Multi-Party Computation for Privacy-Preserving Data Analysis',
        authors: ['Cohen, D.', 'Nakamura, H.', 'Singh, P.'],
        abstract: 'This research introduces novel secure multi-party computation protocols for privacy-preserving data analysis. Our approach enables multiple parties to jointly analyze sensitive data without revealing their private inputs.',
        published_date: '2022-08-12',
        categories: ['cs.CR', 'cs.DC'],
        keywords: ['security', 'privacy', 'cryptography', 'multi-party computation', 'data analysis']
      },
      {
        title: 'Transformer Architectures for Code Generation and Understanding',
        authors: ['Park, S.', 'Gonzalez, M.', 'Almeida, J.'],
        abstract: 'We present specialized transformer architectures for code generation and understanding tasks. Our models achieve state-of-the-art performance on code completion, translation, and bug detection benchmarks.',
        published_date: '2023-01-30',
        categories: ['cs.SE', 'cs.CL', 'cs.LG'],
        keywords: ['transformers', 'code generation', 'program synthesis', 'software engineering', 'deep learning']
      },
      {
        title: 'Distributed Systems for Edge Computing: Challenges and Solutions',
        authors: ['Li, W.', 'Fernandez, E.', 'Sharma, R.'],
        abstract: 'This paper addresses key challenges in designing distributed systems for edge computing environments. We propose novel architectures that handle network heterogeneity, resource constraints, and fault tolerance in edge deployments.',
        published_date: '2022-10-18',
        categories: ['cs.DC', 'cs.NI'],
        keywords: ['distributed systems', 'edge computing', 'fog computing', 'iot', 'networking']
      }
    ];
    
    // Filter papers that match keywords in the query
    let filteredPapers = paperTemplates.filter(paper => {
      return keywords.some(keyword => 
        paper.title.toLowerCase().includes(keyword) || 
        paper.abstract.toLowerCase().includes(keyword) ||
        paper.keywords.some(k => k.includes(keyword)) ||
        paper.authors.some(a => a.toLowerCase().includes(keyword))
      );
    });
    
    // If no matches, return a random selection
    if (filteredPapers.length === 0) {
      filteredPapers = paperTemplates.sort(() => 0.5 - Math.random()).slice(0, 5);
    }
    
    // Apply category filter if specified
    if (categories.length > 0) {
      filteredPapers = filteredPapers.filter(paper => 
        paper.categories.some(c => categories.includes(c))
      );
    }
    
    // Apply date range filter
    if (dateRange !== 'all') {
      const now = new Date();
      let cutoffDate;
      
      switch (dateRange) {
        case '1w':
          cutoffDate = new Date(now.setDate(now.getDate() - 7));
          break;
        case '1m':
          cutoffDate = new Date(now.setMonth(now.getMonth() - 1));
          break;
        case '3m':
          cutoffDate = new Date(now.setMonth(now.getMonth() - 3));
          break;
        case '1y':
          cutoffDate = new Date(now.setFullYear(now.getFullYear() - 1));
          break;
        case '3y':
          cutoffDate = new Date(now.setFullYear(now.getFullYear() - 3));
          break;
        default:
          cutoffDate = new Date(0); // Beginning of time
      }
      
      filteredPapers = filteredPapers.filter(paper => 
        new Date(paper.published_date) >= cutoffDate
      );
    }
    
    // Apply sorting
    switch (sortBy) {
      case 'date-desc':
        filteredPapers.sort((a, b) => new Date(b.published_date) - new Date(a.published_date));
        break;
      case 'date-asc':
        filteredPapers.sort((a, b) => new Date(a.published_date) - new Date(b.published_date));
        break;
      case 'citations':
        // Add random citation counts for demo
        filteredPapers.forEach(paper => {
          paper.citation_count = Math.floor(Math.random() * 1000);
        });
        filteredPapers.sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0));
        break;
      case 'relevance':
      default:
        // For demo, we'll just randomize a bit to simulate relevance ranking
        filteredPapers.sort(() => 0.5 - Math.random());
    }
    
    // Add IDs and URLs
    return filteredPapers.map((paper, index) => ({
      ...paper,
      id: `paper-${Date.now()}-${index}`,
      arxiv_id: `2304.${10000 + Math.floor(Math.random() * 9999)}`,
      pdf_url: `https://arxiv.org/pdf/2304.${10000 + Math.floor(Math.random() * 9999)}.pdf`,
      citation_count: paper.citation_count || Math.floor(Math.random() * 500)
    }));
  };
  
  return (
    <PaperSearchContext.Provider
      value={{
        papers,
        loading,
        error,
        searchPapers
      }}
    >
      {children}
    </PaperSearchContext.Provider>
  );
};
