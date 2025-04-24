import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  TextField,
  InputAdornment,
  Tooltip,
  Tabs,
  Tab
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import InfoIcon from '@mui/icons-material/Info';
import { getConcepts, getConceptRelations } from '../services/api';
import { Concept, ConceptRelation } from '../types/concept';
import VisualizationContainer from './visualizations/VisualizationContainer';

const ConceptVisualization: React.FC = () => {
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [relations, setRelations] = useState<ConceptRelation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [conceptsData, relationsData] = await Promise.all([
          getConcepts(),
          getConceptRelations()
        ]);
        setConcepts(conceptsData);
        setRelations(relationsData);
      } catch (err) {
        console.error('Error fetching concept data:', err);
        setError('Failed to load concept data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const filteredConcepts = concepts.filter(concept =>
    concept.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getRelatedConcepts = (conceptId: number) => {
    return relations.filter(
      relation => relation.source_concept_id === conceptId || relation.target_concept_id === conceptId
    );
  };

  const handleConceptClick = (concept: Concept) => {
    setSelectedConcept(concept);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box className="concept-visualization-container">
      <Typography variant="h4" gutterBottom>
        Concept Visualization
      </Typography>

      <Paper sx={{ mb: 2 }}>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab label="List View" />
          <Tab label="Interactive Visualizations" />
        </Tabs>
      </Paper>

      {activeTab === 0 ? (
        <>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Search concepts..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ mb: 2 }}
          />

          <Box sx={{ display: 'flex', height: 'calc(100vh - 250px)' }}>
            {/* Concept List */}
            <Paper sx={{ width: 300, overflow: 'auto', mr: 2 }}>
              <List>
                {filteredConcepts.length > 0 ? (
                  filteredConcepts.map((concept) => (
                    <React.Fragment key={concept.concept_id}>
                      <ListItem
                        button
                        onClick={() => handleConceptClick(concept)}
                        selected={selectedConcept?.concept_id === concept.concept_id}
                      >
                        <ListItemText primary={concept.name} />
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))
                ) : (
                  <ListItem>
                    <ListItemText primary="No concepts found" />
                  </ListItem>
                )}
              </List>
            </Paper>

            {/* Concept Details and Relations */}
            <Paper sx={{ flex: 1, p: 2, overflow: 'auto' }}>
              {selectedConcept ? (
                <>
                  <Typography variant="h5" gutterBottom>
                    {selectedConcept.name}
                    <Tooltip title="This concept was extracted from research papers by our AI system">
                      <InfoIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle' }} />
                    </Tooltip>
                  </Typography>

                  <Typography variant="body1" paragraph>
                    {selectedConcept.definition}
                  </Typography>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="h6" gutterBottom>
                    Related Concepts
                  </Typography>

                  {getRelatedConcepts(selectedConcept.concept_id).length > 0 ? (
                    <List>
                      {getRelatedConcepts(selectedConcept.concept_id).map((relation) => {
                        const isSource = relation.source_concept_id === selectedConcept.concept_id;
                        const relatedConceptName = isSource ? relation.target_name : relation.source_name;
                        const relatedConceptId = isSource ? relation.target_concept_id : relation.source_concept_id;
                        const relatedConcept = concepts.find(c => c.concept_id === relatedConceptId);

                        return (
                          <ListItem
                            key={`${relation.source_concept_id}-${relation.target_concept_id}`}
                            button
                            onClick={() => relatedConcept && handleConceptClick(relatedConcept)}
                          >
                            <ListItemText
                              primary={relatedConceptName}
                              secondary={
                                isSource
                                  ? `${selectedConcept.name} ${relation.relation_type} ${relatedConceptName}`
                                  : `${relatedConceptName} ${relation.relation_type} ${selectedConcept.name}`
                              }
                            />
                          </ListItem>
                        );
                      })}
                    </List>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No related concepts found.
                    </Typography>
                  )}
                </>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body1" color="text.secondary">
                    Select a concept to view details
                  </Typography>
                </Box>
              )}
            </Paper>
          </Box>
        </>
      ) : (
        <VisualizationContainer />
      )}
    </Box>
  );
};

export default ConceptVisualization;
