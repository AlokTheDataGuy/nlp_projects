import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Paper
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searched, setSearched] = useState(false);

  const handleSearch = () => {
    if (!query.trim()) return;
    
    setSearched(true);
    
    // Simulate search results
    const mockResults = [
      {
        id: '1',
        title: 'Attention Is All You Need',
        authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
        abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.'
      },
      {
        id: '2',
        title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        authors: ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee', 'Kristina Toutanova'],
        abstract: 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.'
      },
      {
        id: '3',
        title: 'Deep Residual Learning for Image Recognition',
        authors: ['Kaiming He', 'Xiangyu Zhang', 'Shaoqing Ren', 'Jian Sun'],
        abstract: 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.'
      }
    ];
    
    setResults(mockResults);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search arXiv Papers
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Box sx={{ display: 'flex', mb: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search papers..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            sx={{ mr: 1 }}
          />
          <Button
            variant="contained"
            color="primary"
            startIcon={<SearchIcon />}
            onClick={handleSearch}
            disabled={!query.trim()}
          >
            Search
          </Button>
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          Search for computer science papers by title, author, or content.
        </Typography>
      </Paper>
      
      {searched && (
        <Box>
          <Typography variant="h6" gutterBottom>
            {results.length} Results
          </Typography>
          
          <Grid container spacing={3}>
            {results.map((paper) => (
              <Grid item xs={12} key={paper.id}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {paper.title}
                    </Typography>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      {paper.authors.join(', ')}
                    </Typography>
                    <Typography variant="body2">
                      {paper.abstract}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button size="small" color="primary">
                      View Details
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
            
            {results.length === 0 && (
              <Grid item xs={12}>
                <Paper sx={{ p: 3, textAlign: 'center' }}>
                  <Typography>No results found. Try a different search term.</Typography>
                </Paper>
              </Grid>
            )}
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default SearchPage;