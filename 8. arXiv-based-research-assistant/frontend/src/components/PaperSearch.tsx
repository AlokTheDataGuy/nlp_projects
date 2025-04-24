import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  CardActions,
  Grid,
  Link,
  CircularProgress,
  Divider,
  Chip
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { searchPapers } from '../services/api';
import { Paper } from '../types/paper';

const PaperSearch: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const results = await searchPapers(searchQuery);
      setPapers(results);
      setSearched(true);
    } catch (error) {
      console.error('Error searching papers:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const formatCategories = (categories: string) => {
    return categories.split(', ').map((category) => (
      <Chip
        key={category}
        label={category}
        size="small"
        sx={{ mr: 0.5, mb: 0.5 }}
      />
    ));
  };

  return (
    <Box className="paper-search-container">
      <Typography variant="h4" gutterBottom>
        Search ArXiv Papers
      </Typography>

      <Box sx={{ display: 'flex', mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search for computer science papers..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
        />
        <Button
          variant="contained"
          color="primary"
          startIcon={<SearchIcon />}
          onClick={handleSearch}
          disabled={!searchQuery.trim() || loading}
          sx={{ ml: 1 }}
        >
          Search
        </Button>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!loading && searched && papers.length === 0 && (
        <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', my: 4 }}>
          No papers found matching your search criteria.
        </Typography>
      )}

      <Grid container spacing={2}>
        {papers.map((paper) => (
          <Grid sx={{ gridColumn: 'span 12' }} key={paper.paper_id}>
            <Card className="paper-card">
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {paper.title}
                </Typography>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  {paper.authors} • {paper.published_date}
                </Typography>
                <Box sx={{ my: 1 }}>
                  {formatCategories(paper.categories)}
                </Box>
                <Divider sx={{ my: 1 }} />
                <Typography variant="body2" paragraph>
                  {paper.abstract}
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  color="primary"
                  component={Link}
                  href={`https://arxiv.org/abs/${paper.paper_id}`}
                  target="_blank"
                  rel="noopener"
                >
                  View on ArXiv
                </Button>
                <Button
                  size="small"
                  color="primary"
                  component={Link}
                  href={`https://arxiv.org/pdf/${paper.paper_id}`}
                  target="_blank"
                  rel="noopener"
                >
                  View PDF
                </Button>
                {paper.pdf_url && (
                  <Button
                    size="small"
                    color="primary"
                    component={Link}
                    href={paper.pdf_url}
                    target="_blank"
                    rel="noopener"
                  >
                    Download PDF
                  </Button>
                )}
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default PaperSearch;
