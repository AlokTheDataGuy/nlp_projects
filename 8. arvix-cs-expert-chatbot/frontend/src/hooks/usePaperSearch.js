import { useContext } from 'react';
import { PaperSearchContext } from '../context/PaperSearchContext';

/**
 * Hook to access the paper search context
 */
export const usePaperSearch = () => {
  const context = useContext(PaperSearchContext);
  
  if (!context) {
    throw new Error('usePaperSearch must be used within a PaperSearchProvider');
  }
  
  return context;
};
